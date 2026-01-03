import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# -----------------------------------------------------------------------------
# 1. SETUP & MODEL LOADING
# -----------------------------------------------------------------------------
MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"Loading {MODEL_ID} in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

# FIX: Explicitly set the ID if the tokenizer misses it
MASK_TOKEN_ID = tokenizer.mask_token_id
if MASK_TOKEN_ID is None:
    MASK_TOKEN_ID = 126336  # The specific ID for LLaDA-8B

print(f"Using MASK_TOKEN_ID: {MASK_TOKEN_ID}")


def apply_dpp_guidance(logits, alpha=15.0, quality_scale=0.1):
    if logits.shape[0] < 2:
        return logits

        # We detach from the main model graph (which we don't want to update)
    # But we MUST enable grad to calculate the 'force field' relative to these logits
    with torch.enable_grad():
        # Create a new leaf variable that tracks gradients
        logits_in = logits.detach().clone().requires_grad_(True)

        # 1. Softmax to get probability distribution
        probs = torch.softmax(logits_in, dim=-1)

        # 2. Sentence Embeddings (Mean pool)
        sentence_vec = probs.mean(dim=1)

        # 3. Normalize for Cosine Similarity
        norm_vec = F.normalize(sentence_vec, p=2, dim=1)

        # 4. Compute Kernel Matrix
        K = torch.mm(norm_vec, norm_vec.t())

        # 5. Quality Weighting
        max_conf = probs.max(dim=-1).values.mean(dim=1)
        quality_matrix = torch.outer(max_conf, max_conf)

        # 6. The Matrix to Maximize
        L_matrix = K * (1 + quality_scale * quality_matrix)

        # 7. Compute Determinant Loss
        identity = torch.eye(L_matrix.shape[0], device=L_matrix.device)
        loss = -torch.logdet(L_matrix + 1e-4 * identity)

        # 8. Calculate Gradient
        # This will now work because we are inside enable_grad()
        grad = torch.autograd.grad(loss, logits_in)[0]

    # Apply Repulsion (outside the enable_grad block to keep things clean)
    guided_logits = logits - (alpha * grad)

    return guided_logits  # 3. LLaDA GENERATION LOOP


@torch.no_grad()
def generate_llada(
        prompt_text,
        batch_size=4,
        steps=20,
        gen_length=64,
        use_dpp=False,
        dpp_alpha=3.0,  # Set this to your "Sweet Spot" (e.g., 3.0 or 4.0)
        quality_scale=1.0

):
    mask_id = MASK_TOKEN_ID

    messages = [{"role": "user", "content": prompt_text}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = input_ids.repeat(batch_size, 1).to(model.device)

    mask_tokens = torch.full((batch_size, gen_length), mask_id, device=model.device, dtype=torch.long)
    input_ids = torch.cat([input_ids, mask_tokens], dim=1)

    L = input_ids.shape[1]
    prompt_len = L - gen_length

    for step in range(steps):
        outputs = model(input_ids)
        logits = outputs.logits
        gen_logits = logits[:, prompt_len:, :]

        current_alpha = dpp_alpha * (1 - (step / steps))

        if use_dpp and current_alpha > 0.1:
            gen_logits = apply_dpp_guidance(
                gen_logits,
                alpha=current_alpha,
                quality_scale=quality_scale,
            )

        probs = F.softmax(gen_logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)

        confidences = probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)

        progress = (step + 1) / steps
        n_unmasked = int(gen_length * progress)

        if step == steps - 1:
            input_ids[:, prompt_len:] = pred_ids
            break

        noise = torch.rand_like(confidences)
        noisy_conf = confidences + noise

        top_k_vals, top_k_indices = torch.topk(noisy_conf, k=n_unmasked, dim=1)

        new_gen_ids = torch.full_like(pred_ids, mask_id)
        new_gen_ids.scatter_(1, top_k_indices, pred_ids.gather(1, top_k_indices))

        input_ids[:, prompt_len:] = new_gen_ids

    return tokenizer.batch_decode(input_ids[:, prompt_len:], skip_special_tokens=True)

# PROMPT = "Write a haiku about a robot realizing it is alive."
# PROMPT = "Explain a metaphor for how neural networks learn."
# PROMPT = "Write a python function to check if a word is a palindrome"

# PROMPT = "Prove that if n is even, n squared is divisible by 4."

# PROMPT = "Explain what 'Time' is."
# PROMPT = "Write a Python function `def sum_to_n(n):` that calculates the sum of all numbers from 0 to n."

# PROMPT = "Prove the following Lean Statement: `theorem sum_of_evens (n : Nat) : Even (n * (n + 1)) := by`"

PROMPT = "Describe the logic to prove that n*(n+1) is even. Do not use code."

print(f"PROMPT: {PROMPT}")

alpha_vals = [1, 3, 5, 10]
quality_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
print(f"\n--- Baseline (No Diversity) ---")
t = time.time()
base_samples = generate_llada(PROMPT, batch_size=4, steps=32, use_dpp=False)
print(f"Generation time: {time.time() - t:.2f} seconds")

for i, s in enumerate(base_samples):
    print(f"[{i + 1}] {s.strip().replace(chr(10), ' / ')}")

for alpha in alpha_vals:
    for quality_scale in quality_scales:
        print(f"\n--- DPP Guided (Alpha={alpha}, Quality Scale={quality_scale}) ---")
        t = time.time()
        dpp_samples = generate_llada(PROMPT, batch_size=4, steps=32, use_dpp=True, dpp_alpha=alpha,
                                     quality_scale=quality_scale)
        print(f"Generation time: {time.time() - t:.2f} seconds")
        for i, s in enumerate(dpp_samples):
            print(f"[{i + 1}] {s.strip().replace(chr(10), ' / ')}")
