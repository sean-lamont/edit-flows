import time
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# -----------------------------------------------------------------------------
# 1. SETUP
# -----------------------------------------------------------------------------
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

MASK_TOKEN_ID = tokenizer.mask_token_id
if MASK_TOKEN_ID is None:
    MASK_TOKEN_ID = 126336

# Extract Embedding Matrix for Semantic Guidance
if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
    EMBEDDING_MATRIX = model.model.embed_tokens.weight
else:
    EMBEDDING_MATRIX = None
    print("Warning: Embedding matrix not found. 'embeddings' target will fail.")


def apply_dpp_guidance(
        logits,
        alpha=3.0,
        quality_scale=1.0,
        pooling_method="mean",
        use_projection=True,
        kernel_target="logits",
        loss_type="diverseflow"  # 'volume' (Old) or 'diverseflow' (Correct)
):
    if logits.shape[0] < 2: return logits

    with torch.enable_grad():
        logits_in = logits.detach().clone().requires_grad_(True)
        probs = torch.softmax(logits_in, dim=-1)

        # ... (Feature Extraction & Pooling same as before) ...
        if kernel_target == "embeddings" and EMBEDDING_MATRIX is not None:
            W = EMBEDDING_MATRIX.to(probs.device).detach()
            features = torch.matmul(probs, W)
        else:
            features = probs

        if pooling_method == "max":
            batch_vecs = features.max(dim=1).values
        else:
            batch_vecs = features.mean(dim=1)

        # Kernel Calculation
        norm_vec = F.normalize(batch_vecs, p=2, dim=1)
        K = torch.mm(norm_vec, norm_vec.t())

        # Quality & Jitter
        max_conf = probs.max(dim=-1).values.mean(dim=1)
        quality_matrix = torch.outer(max_conf, max_conf)
        identity = torch.eye(K.shape[0], device=K.device)
        jitter = 1e-4  # Tiny jitter for stability

        # The Kernel Matrix L
        L = K * (1 + quality_scale * quality_matrix)

        # --- THE FIX: DIVERSEFLOW LOSS ---
        if loss_type == "diverseflow":
            # Maximize P(Batch) = det(L) / det(L+I)
            # Minimize Loss = - ( logdet(L) - logdet(L+I) )

            # We add jitter to both for numerical safety
            term1 = torch.logdet(L + jitter * identity)
            term2 = torch.logdet(L + identity + jitter * identity)
            loss = -(term1 - term2)
        else:
            # Old "Volume Maximization"
            loss = -torch.logdet(L + jitter * identity)

        grad = torch.autograd.grad(loss, logits_in)[0]

    # ... (Projection & Update same as before) ...
    g_norm = torch.norm(grad, p=2, dim=-1, keepdim=True)
    grad_safe = grad / (g_norm + 1e-8)

    if use_projection:
        u = logits.detach()
        inner_prod = (grad_safe * u).sum(dim=-1, keepdim=True)
        u_norm_sq = (u * u).sum(dim=-1, keepdim=True)
        proj = (inner_prod / (u_norm_sq + 1e-8)) * u
        grad_safe = grad_safe - proj

    return logits - (alpha * grad_safe)
# -----------------------------------------------------------------------------
# 3. GENERATION LOOP
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_llada(
        prompt,
        batch_size=4,
        steps=32,
        # Config Dictionary Unpacking
        use_dpp=False,
        alpha=3.0,
        quality=1.0,
        pool="mean",
        proj=True,
        target="logits"
):
    mask_id = MASK_TOKEN_ID

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = input_ids.repeat(batch_size, 1).to(model.device)

    gen_len = 64
    mask_tokens = torch.full((batch_size, gen_len), mask_id, device=model.device, dtype=torch.long)
    input_ids = torch.cat([input_ids, mask_tokens], dim=1)

    prompt_len = input_ids.shape[1] - gen_len

    for step in range(steps):
        outputs = model(input_ids)
        gen_logits = outputs.logits[:, prompt_len:, :]

        if use_dpp:
            # Linear Decay Schedule
            curr_alpha = alpha * (1 - (step / steps))
            if curr_alpha > 0.1:
                gen_logits = apply_dpp_guidance(
                    gen_logits,
                    alpha=curr_alpha,
                    quality_scale=quality,
                    pooling_method=pool,
                    use_projection=proj,
                    kernel_target=target
                )

        probs = F.softmax(gen_logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)

        confidences = probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)
        progress = (step + 1) / steps
        n_unmasked = int(gen_len * progress)

        if step == steps - 1:
            input_ids[:, prompt_len:] = pred_ids
            break

        noise = torch.rand_like(confidences)
        top_k = torch.topk(confidences + noise, k=n_unmasked, dim=1)

        new_ids = torch.full_like(pred_ids, mask_id)
        new_ids.scatter_(1, top_k.indices, pred_ids.gather(1, top_k.indices))
        input_ids[:, prompt_len:] = new_ids

    return tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)

PROMPT = "Describe the logic to prove that n*(n+1) is even. Do not use code."
# PROMPT = "Write a python function to check if a word is a palindrome"
# PROMPT = "Write a haiku about a robot realizing it is alive."
# PROMPT = "Explain what 'Time' is."
# PROMPT = "Explain a metaphor for how neural networks learn."
# PROMPT = "Write a python program to train a neural network"

print(f"PROMPT: {PROMPT}\n")

# Define your experiments here
settings = [
    {
        "name": "Baseline (No DPP)",
        "use_dpp": False,
        "alpha": 0.0, "quality": 0.0, "pool": "mean", "proj": False, "target": "logits"
    },
    {
        "name": "DPP (Logits, Alpha=3.0, No Projection)",
        "use_dpp": True,
        "alpha": 3.0, "quality": 1.0, "pool": "mean", "proj": False, "target": "logits"
    },
    {
        "name": "DPP (Logits, Alpha=3.0, Projected)",
        "use_dpp": True,
        "alpha": 3.0, "quality": 1.0, "pool": "mean", "proj": True, "target": "logits"
    },
    # {
    #     "name": "DPP (Semantic, Alpha=5.0, Projected)",
    #     "use_dpp": True,
    #     "alpha": 3.0, "quality": 1.0, "pool": "mean", "proj": True, "target": "embeddings"
    # }
]

for cfg in settings:
    print(f"--- {cfg['name']} ---")
    start = time.time()

    samples = generate_llada(
        PROMPT,
        batch_size=4,
        steps=32,
        use_dpp=cfg['use_dpp'],
        alpha=cfg['alpha'],
        quality=cfg['quality'],
        pool=cfg['pool'],
        proj=cfg['proj'],
        target=cfg['target']
    )

    print(f"Time: {time.time() - start:.2f}s")
    for i, s in enumerate(samples):
        print(f"[{i + 1}] {s.strip().replace(chr(10), ' / ')}")
    print("")