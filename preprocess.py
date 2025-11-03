import torch
import glob
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
import datasets

# (Imports and arg parsing are the same...)
try:
    from utils import opt_align_xs_to_zs
except ImportError:
    print("Error: Could not import 'opt_align_xs_to_zs' from 'utils.py'.")
    exit(1)

# (Constants are the same...)
GAP_TOKEN = 151651
INITIAL_ATTEMPT_STRING = "Initial Attempt"


def parse_args():
    # (This function is unchanged)
    parser = argparse.ArgumentParser(description="Pre-process Goedel dataset for edit flows.")

    # --- Paths ---
    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed_data_code_only",
        help="Directory containing the original .jsonl files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="precomputed_hf_dataset",
        help="Directory to save the new Hugging Face dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the Hugging Face tokenizer (e.g., 'gpt2', 'bert-base-uncased')."
    )

    # --- Filtering Thresholds (from datamodule.py) ---
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=2000,
        help="Max token length for context. Samples longer than this are truncated."
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=4000,
        help="Max token length for prev_attempt (x0) and target (x1). "
             "Samples with *original* lengths longer than this are FILTERED."
    )
    parser.add_argument(
        "--max_z_len",
        type=int,
        default=8000,
        help="Max token length for aligned z0/z1. "
             "Samples with *aligned* lengths longer than this are FILTERED."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # (Setup, tokenizer loading, and file finding are unchanged...)
    print(f"Starting pre-processing...")
    print(f"  Input data:  '{args.data_dir}'")
    print(f"  Output data: '{args.output_dir}'")
    print(f"  Tokenizer:   '{args.model_name}'")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            print("Warning: tokenizer.pad_token_id is None. Setting to eos_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            print("Error: tokenizer.pad_token_id and tokenizer.eos_token_id are both None.")
            exit(1)

    PAD_TOKEN_ID = tokenizer.pad_token_id

    files = glob.glob(f'{args.data_dir}/*.jsonl')
    if not files:
        print(f"Error: No .jsonl files found in '{args.data_dir}'.")
        exit(1)

    print(f"Found {len(files)} .jsonl files.")

    total_samples = 0
    filtered_samples = 0
    all_good_samples = []

    for file_path in tqdm(files, desc="Processing files"):
        with open(file_path, 'r') as f:
            for line in f:
                total_samples += 1

                try:
                    sample = json.loads(line)

                    # (Filtering and tokenizing are unchanged...)
                    if len(sample['target']) <= 10:
                        filtered_samples += 1
                        continue

                    if len(sample.get('prev_attempt', '')) < 2:
                        sample['prev_attempt'] = INITIAL_ATTEMPT_STRING

                    context = sample['context']
                    prev_attempt = sample['prev_attempt']
                    target = sample['target']

                    context_ids = tokenizer(context, return_tensors='pt').input_ids.squeeze(0)
                    prev_ids = tokenizer(prev_attempt, return_tensors='pt').input_ids.squeeze(0)
                    target_ids = tokenizer(target, return_tensors='pt').input_ids.squeeze(0)

                    if prev_ids.shape[0] > args.max_len or target_ids.shape[0] > args.max_len:
                        filtered_samples += 1
                        continue

                    context_ids = context_ids[:args.max_context_len]

                    # (Alignment and z-filtering are unchanged...)
                    z0, z1 = opt_align_xs_to_zs(
                        prev_ids.unsqueeze(0),
                        target_ids.unsqueeze(0),
                        PAD_TOKEN_ID,
                        GAP_TOKEN
                    )
                    z0 = z0.squeeze(0)
                    z1 = z1.squeeze(0)

                    if z0.shape[0] > args.max_z_len:
                        filtered_samples += 1
                        continue

                    # --- MODIFIED: Added 'x0' to the saved data ---
                    output_data = {
                        'x0': prev_ids.long().tolist(),  # <--- NEW: Save original x0
                        'z0': z0.long().tolist(),
                        'z1': z1.long().tolist(),
                        'x1': target_ids.long().tolist(),
                        'context': context_ids.long().tolist(),
                        'idx': sample.get('idx', total_samples - 1),
                        'type': sample.get('type', 'unknown')
                    }
                    all_good_samples.append(output_data)

                except Exception as e:
                    print(f"\n[Warning] Failed to process line {total_samples} in {file_path}. Error: {e}")
                    filtered_samples += 1

    # (Dataset creation and saving are unchanged...)
    print("\n--- Processing Complete ---")
    print(f"Total samples found:     {total_samples}")
    print(f"Samples filtered (OOM/Error): {filtered_samples}")
    print(f"Samples to save:         {len(all_good_samples)}")

    if all_good_samples:
        print("Creating Hugging Face Dataset...")
        hf_dataset = datasets.Dataset.from_list(all_good_samples)

        print(f"Saving dataset to disk at '{args.output_dir}'...")
        hf_dataset.save_to_disk(args.output_dir)
        print("Done.")
    else:
        print("No samples passed the filters. No dataset saved.")


if __name__ == "__main__":
    main()