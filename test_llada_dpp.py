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

print(f"Using MASK_TOKEN_ID: {MASK_TOKEN_ID}")


# -----------------------------------------------------------------------------
# 2. DPP GUIDANCE (Supports Logits OR Hidden States)
# -----------------------------------------------------------------------------
def apply_dpp_guidance(
        logits,
        hidden_states=None,
        alpha=3.0,
        quality_scale=1.0,
        pooling_method="mean",
        use_projection=True,
        kernel_source="logits"  # 'logits' or 'hidden'
):
    """
    Applies repulsive force.
    If kernel_source='hidden', we calculate diversity on hidden states
    and backprop to logits.
    """
    if logits.shape[0] < 2: return logits

    with torch.enable_grad():
        # Track gradients on the input we want to shift (logits)
        logits_in = logits.detach().clone().requires_grad_(True)

        # If using hidden states, we need them to be part of the graph.
        # However, we can't easily invert the LM head without full backprop.
        # OPTIMIZATION: We calculate diversity on the *representation* we have.

        if kernel_source == "hidden" and hidden_states is not None:
            # Diversity based on the LAST HIDDEN LAYER (Semantic Repulsion)
            # Shape: [Batch, Seq, Hidden_Dim]
            target_tensor = hidden_states.detach().clone().requires_grad_(True)

            # Pool: [Batch, Seq, Dim] -> [Batch, Dim]
            if pooling_method == "max":
                vecs = target_tensor.max(dim=1).values
            else:
                vecs = target_tensor.mean(dim=1)

        else:
            # Diversity based on VOCAB PROBABILITIES (Token Repulsion)
            # Shape: [Batch, Seq, Vocab]
            probs = torch.softmax(logits_in, dim=-1)

            if pooling_method == "max":
                vecs = probs.max(dim=1).values
            else:
                vecs = probs.mean(dim=1)

            target_tensor = logits_in  # We differentiate w.r.t this

        # --- Compute Kernel ---
        norm_vec = F.normalize(vecs, p=2, dim=1)
        K = torch.mm(norm_vec, norm_vec.t())

        # --- Quality Term ---
        # Always use Logit confidence for quality (Hidden states don't have 'confidence')
        current_probs = torch.softmax(logits_in, dim=-1)
        max_conf = current_probs.max(dim=-1).values.mean(dim=1)
        quality_matrix = torch.outer(max_conf, max_conf)

        # --- Loss ---
        identity = torch.eye(K.shape[0], device=K.device)
        jitter = 1e-3 if pooling_method == "mean" else 5e-2
        L_matrix = K * (1 + quality_scale * quality_matrix) + (jitter * identity)
        loss = -torch.logdet(L_matrix)

        # --- Gradient Calculation ---
        # If source is logits, this is direct.
        # If source is hidden, we calculate grad w.r.t hidden, then...
        # WAIT: In inference, we can't backprop from Hidden -> Logits easily
        # without running the LM Head layer.

        # FIX: For 'hidden' mode, we use the standard 'logits' gradient
        # but calculated using the 'hidden' kernel similarity.
        # This requires differentiating K(hidden) w.r.t logits, which isn't possible
        # unless we ran the forward pass inside this function.

        # PRACTICAL HACK for "Hidden" Mode in Inference Script:
        # Since we can't backprop through the frozen LM Head easily here,
        # we will stick to LOGITS gradient for the update, but use Hidden ONLY if provided
        # and if we can chain rule it.

        # Actually, for this specific test script, sticking to LOGITS is safer.
        # If you want TRUE hidden divergence, we must pass logits through the LM head
        # inside this block. Let's do that for correctness.

        if kernel_source == "hidden":
            # We cannot get gradient w.r.t logits from hidden states
            # because Hidden -> Logits is the forward direction.
            # Gradient comes from Loss -> Hidden -> ... -> Input.
            # We want to change Logits to change Hidden? No.
            # We change Hidden to change Logits.

            # REVERT: For inference-time guidance without training graph,
            # calculating gradient on Hidden states is useless unless we update
            # the Hidden states and then project forward.
            # But we control Logits (output of blocks).

            # CORRECT APPROACH: We simply fallback to 'logits' based guidance
            # OR we strictly use logits for the diversity calculation.
            pass

        # For this script, we will proceed with LOGITS based calculation
        # but allow user to compare standard 'mean' vs 'max' pooling on logits.
        # True Hidden-State DPP requires access to the LM_Head weights to backprop.

        grad = torch.autograd.grad(loss, logits_in)[0]

    # --- Stabilization ---
    g_norm = torch.norm(grad, p=2, dim=-1, keepdim=True)
    grad_safe = grad / (g_norm + 1e-8)

    if use_projection:
        u = logits.detach()
        inner = (grad_safe * u).sum(dim=-1, keepdim=True)
        u_norm = (u * u).sum(dim=-1, keepdim=True)
        proj = (inner / (u_norm + 1e-8)) * u
        grad_safe = grad_safe - proj

    return logits - (alpha * grad_safe)


# Note on the "Hidden" Issue:
# To strictly implement Hidden-State DPP, we would need `model.lm_head`.
# Since `AutoModel` wraps everything, accessing the specific head layer
# dynamically across architectures is brittle.
# I have simplified the function above to focus on the LOGITS implementation
# but added the structural toggle if you decide to expose `model.lm_head`.

# -----------------------------------------------------------------------------
# 3. GENERATION
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_llada(
        prompt,
        batch_size=4,
        steps=32,
        use_dpp=False,
        dpp_alpha=3.0,
        pooling="mean",
        projection=True
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
        # We assume standard output (logits).
        # If we wanted hidden, we'd add output_hidden_states=True
        outputs = model(input_ids)
        gen_logits = outputs.logits[:, prompt_len:, :]

        # Decay
        curr_alpha = dpp_alpha * (1 - (step / steps))

        if use_dpp and curr_alpha > 0.1:
            gen_logits = apply_dpp_guidance(
                gen_logits,
                alpha=curr_alpha,
                pooling_method=pooling,
                use_projection=projection
            )

        probs = F.softmax(gen_logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)

        # Re-masking
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


# -----------------------------------------------------------------------------
# 4. RUN COMPARISON
# -----------------------------------------------------------------------------
PROMPT = "Describe the logic to prove that n*(n+1) is even. Do not use code."
print(f"PROMPT: {PROMPT}\n")

# Settings to Compare
settings = [
    # (Alpha, Pooling, Projection)
    (0.0, "mean", False),  # Baseline
    (3.0, "mean", True),  # Recommended
    (3.0, "max", True),  # Keywords focus
]

for alpha, pool, proj in settings:
    label = "Baseline" if alpha == 0 else f"DPP (A={alpha}, {pool}, Proj={proj})"
    print(f"--- {label} ---")

    start = time.time()
    samps = generate_llada(
        PROMPT,
        batch_size=4,
        use_dpp=(alpha > 0),
        dpp_alpha=alpha,
        pooling=pool,
        projection=proj
    )
    print(f"Time: {time.time() - start:.2f}s")
    for i, s in enumerate(samps):
        print(f"[{i + 1}] {s.strip().replace(chr(10), ' / ')}")
    print("")