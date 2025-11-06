import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Optional, Tuple


def get_model_and_tokenizer_info(
    base_model_id: str,
    lora_adapter_id: Optional[str] = None,
    special_token: str = "<GAP>",
    torch_dtype=torch.float16,
) -> Tuple[AutoTokenizer, int, int]:
    """
    Loads a tokenizer and model, optionally applies LoRA adapters, checks vocabulary size,
    and ensures a special token exists, adding it if necessary.

    Args:
        base_model_id (str): The Hugging Face identifier for the base model.
        lora_adapter_id (Optional[str]): The Hugging Face identifier for the LoRA adapters.
        special_token (str): The special token to check for (e.g., "<GAP>").
        torch_dtype: The torch dtype to use for loading the model (e.g., torch.float16).

    Returns:
        A tuple containing:
        - tokenizer (AutoTokenizer): The loaded and configured tokenizer.
        - special_token_id (int): The integer ID of the special token.
        - full_vocab_size (int): The true vocabulary size from the model's embedding matrix.
    """
    print(f"--- Loading Model & Tokenizer ---")
    print(f"  Base Model: {base_model_id}")
    if lora_adapter_id:
        print(f"  LoRA Adapter: {lora_adapter_id}")

    # 1. Load base model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_adapter_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            dtype=torch_dtype,
            device_map="auto"
        )

        # 2. Apply LoRA adapters if provided
        if lora_adapter_id:
            print(f"  Applying LoRA adapters...")
            model = PeftModel.from_pretrained(model, lora_adapter_id)

        print("✅ Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        print("   Please ensure you are logged into huggingface-cli and have access to the model(s).")
        raise


    # 4. Check for and add the special token if it doesn't exist
    print(f"\n🔧 Special Token Setup ('{special_token}'):")
    if special_token not in tokenizer.get_vocab():
        print(f"   ⚠️ Special token '{special_token}' not found. Adding it.")
        tokenizer.add_special_tokens({'additional_special_tokens': [special_token]})
    else:
        print(f"   ✅ Special token '{special_token}' already exists.")

    print(f"\n🔎 Vocabulary Size Check:")

    # 3. Compare vocab sizes to find the true vocabulary size
    tokenizer_vocab_size = len(tokenizer)
    model_embedding_size = model.get_input_embeddings().weight.size(0)

    full_vocab_size = max(len(tokenizer), model_embedding_size)

    print(f"   - Tokenizer vocabulary size: {tokenizer_vocab_size}")
    print(f"   - Model embedding matrix size: {model_embedding_size}")

    if tokenizer_vocab_size < model_embedding_size:
        print(f"   ⚠️ Mismatch detected! Using model's embedding size as the true 'full_vocab_size': {full_vocab_size}")

    special_token_id = tokenizer.convert_tokens_to_ids(special_token)
    print(f"   '{special_token}' ID is: {special_token_id}")

    return tokenizer, special_token_id, full_vocab_size


if __name__ == "__main__":
    print("--- Example 1: Loading a model with LoRA adapters ---")
    repair_llama_tokenizer, repair_llama_gap_id, repair_llama_vocab_size = get_model_and_tokenizer_info(
        base_model_id="TheBloke/CodeLlama-7B-fp16",
        lora_adapter_id="ASSERT-KTH/RepairLLaMA-IR1-OR1",
        special_token="<GAP>",
        torch_dtype=torch.float16
    )
    print("\n--- Results for RepairLLaMA ---")
    print(f"  Tokenizer: {repair_llama_tokenizer.__class__.__name__}")
    print(f"  <GAP> Token ID: {repair_llama_gap_id}")
    print(f"  Full Vocab Size: {repair_llama_vocab_size}")

    print("\n" + "="*50 + "\n")

    print("--- Example 2: Loading a base model directly ---")
    goedel_tokenizer, goedel_gap_id, goedel_vocab_size = get_model_and_tokenizer_info(
        base_model_id="Goedel-LM/Goedel-Prover-V2-8B",
        special_token="<|quad_end|>",
        torch_dtype=torch.bfloat16 # Goedel model works well with bfloat16
    )
    print("\n--- Results for Goedel-Prover ---")
    print(f"  Tokenizer: {goedel_tokenizer.__class__.__name__}")
    print(f"  <GAP> Token ID: {goedel_gap_id}")
    print(f"  Full Vocab Size: {goedel_vocab_size}")