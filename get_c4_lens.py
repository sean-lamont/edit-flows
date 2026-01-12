import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATASET_NAME = "c4"
SUBSET = "realnewslike"
TOKENIZER_ID = "bert-base-uncased"  # Change to "meta-llama/Llama-2-7b-hf" or similar if needed
MAX_CAP = 1024
NUM_SAMPLES = 50_000  # 50k is usually enough for a statistically significant mean


def main():
    print(f"Loading tokenizer: {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # Streaming mode prevents downloading terabytes of data
    print(f"Streaming {DATASET_NAME}/{SUBSET}...")
    dataset = load_dataset(DATASET_NAME, SUBSET, split="train", streaming=True)

    total_capped_length = 0
    count = 0

    print(f"Calculating average length (capped at {MAX_CAP}) over {NUM_SAMPLES} samples...")

    # Iterate through the stream
    for i, sample in tqdm(enumerate(dataset), total=NUM_SAMPLES):
        if i >= NUM_SAMPLES:
            break

        text = sample['text']

        # Tokenize without truncation first to get the real length
        token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        raw_len = len(token_ids)

        # Apply the logic: if len >= 1024, count as 1024
        capped_len = min(raw_len, MAX_CAP)

        total_capped_length += capped_len
        count += 1

    avg_length = total_capped_length / count

    print("\n" + "=" * 30)
    print(f"RESULTS ({SUBSET})")
    print("=" * 30)
    print(f"Samples Processed: {count}")
    print(f"Max Length Cap:    {MAX_CAP}")
    print(f"Average Length:    {avg_length:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()