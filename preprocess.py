import torch
import glob
import json
import os
import argparse
import multiprocessing
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset

# CRITICAL: Assumes 'utils.py' is in the same directory or Python path
# so that worker processes can import it.
try:
    from utils import opt_align_xs_to_zs
except ImportError:
    print("Error: Could not import 'opt_align_xs_to_zs' from 'utils.py'.")
    print("Please make sure 'preprocess.py' is in the same directory as 'utils.py'.")
    exit(1)

# --- Constants from your files ---
GAP_TOKEN = 151651
INITIAL_ATTEMPT_STRING = "Initial Attempt"

# --- Global tokenizer for worker processes ---
# This will be initialized once per worker in init_worker
global_tokenizer = None


def init_worker(model_name: str):
    """
    Initializer function for each worker process in the pool.
    Loads the tokenizer into a global variable for that process.
    """
    global global_tokenizer
    print(f"Initializing worker (PID: {os.getpid()})...")
    global_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if global_tokenizer.pad_token_id is None:
        if global_tokenizer.eos_token_id is not None:
            global_tokenizer.pad_token_id = global_tokenizer.eos_token_id
        else:
            raise ValueError(f"Tokenizer {model_name} has no pad_token_id or eos_token_id.")


def process_file(file_path: str, args: argparse.Namespace):
    """
    This function is run by a single worker process.
    It processes one entire .jsonl file and returns a list of "good" samples.
    """
    global global_tokenizer

    # This import happens *inside* the worker
    from utils import opt_align_xs_to_zs

    tokenizer = global_tokenizer
    PAD_TOKEN_ID = tokenizer.pad_token_id

    good_samples_for_this_file = []
    line_count = 0

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line_count += 1
                try:
                    sample = json.loads(line)

                    # --- Apply original GoedelDataset filter (from goedel_dataset.py) ---
                    if len(sample['target']) <= 10:
                        continue

                    if len(sample.get('prev_attempt', '')) < 2:
                        sample['prev_attempt'] = INITIAL_ATTEMPT_STRING

                    # --- Tokenize (from datamodule.py) ---
                    context = sample['context']
                    prev_attempt = sample['prev_attempt']
                    target = sample['target']

                    context_ids = tokenizer(context, return_tensors='pt').input_ids.squeeze(0)
                    prev_ids = tokenizer(prev_attempt, return_tensors='pt').input_ids.squeeze(0)
                    target_ids = tokenizer(target, return_tensors='pt').input_ids.squeeze(0)

                    # --- Check Original Length Filters (x0, x1) ---
                    if prev_ids.shape[0] > args.max_len or target_ids.shape[0] > args.max_len:
                        continue

                    # --- Truncate context (from datamodule.py) ---
                    context_ids = context_ids[:args.max_context_len]

                    # --- Run Alignment (from collate.py) ---
                    # This is the most expensive CPU step
                    z0, z1 = opt_align_xs_to_zs(
                        prev_ids.unsqueeze(0),
                        target_ids.unsqueeze(0),
                        PAD_TOKEN_ID,
                        GAP_TOKEN
                    )
                    z0 = z0.squeeze(0)
                    z1 = z1.squeeze(0)

                    # --- Check Aligned Length Filter (z) ---
                    if z0.shape[0] > args.max_z_len:
                        continue

                    # --- This sample is "good". Add its data to the list. ---
                    output_data = {
                        'x0': prev_ids.long().tolist(),
                        'z0': z0.long().tolist(),
                        'z1': z1.long().tolist(),
                        'x1': target_ids.long().tolist(),
                        'context': context_ids.long().tolist(),
                        'idx': sample.get('idx', line_count - 1),
                        'type': sample.get('type', 'unknown')
                    }
                    good_samples_for_this_file.append(output_data)

                except Exception as e:
                    # Log error for the specific line but continue processing
                    print(
                        f"\n[Warning] (PID: {os.getpid()}) Failed to process line {line_count} in {file_path}. Error: {e}")

    except Exception as e:
        print(f"\n[Error] (PID: {os.getpid()}) Failed to open or read {file_path}. Error: {e}")

    return good_samples_for_this_file


def parse_args():
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
        default="Goedel-LM/Goedel-Prover-V2-8B",
        help="Name of the Hugging Face tokenizer (e.g., 'gpt2', 'bert-base-uncased')."
    )

    # --- Filtering Thresholds ---
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
        help="Max token length for prev_attempt (x0) and target (x1). Samples longer are FILTERED."
    )
    parser.add_argument(
        "--max_z_len",
        type=int,
        default=8000,
        help="Max token length for aligned z0/z1. Samples longer are FILTERED."
    )

    # --- NEW: Multiprocessing ---
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of CPU processes to use. Defaults to all available cores."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_workers is None:
        args.num_workers = os.cpu_count()
        print(f"--num_workers not set, defaulting to {args.num_workers} cores.")

    print(f"Starting pre-processing with {args.num_workers} workers...")
    print(f"  Input data:  '{args.data_dir}'")
    print(f"  Output data: '{args.output_dir}'")
    print(f"  Tokenizer:   '{args.model_name}'")

    # 1. Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Find all .jsonl files
    files = glob.glob(f'{args.data_dir}/*.jsonl')
    if not files:
        print(f"Error: No .jsonl files found in '{args.input_dir}'.")
        exit(1)

    print(f"Found {len(files)} .jsonl files to process.")

    # 3. Process files in parallel
    all_good_samples = []

    # Create a partial function to pass the 'args' to our process_file function
    # The tokenizer will be handled by the worker initializer
    process_func = partial(process_file, args=args)

    print("Starting process pool...")
    with multiprocessing.Pool(args.num_workers, initializer=init_worker, initargs=(args.model_name,)) as pool:

        # Use imap_unordered for memory efficiency and to get results as they complete
        # Wrap with tqdm to show progress (progress is per file, not per line)
        for good_samples_list in tqdm(pool.imap_unordered(process_func, files), total=len(files),
                                      desc="Processing files"):
            all_good_samples.extend(good_samples_list)

    print("\n--- Processing Complete ---")
    total_found = sum([1 for f in files for line in open(f)])  # A rough way to count
    print(f"Total samples found (approx): {total_found}")
    print(f"Samples saved (post-filter):  {len(all_good_samples)}")

    # 4. Create and save the Hugging Face dataset
    if all_good_samples:
        print("Creating Hugging Face Dataset...")
        hf_dataset = Dataset.from_list(all_good_samples)

        print(f"Saving dataset to disk at '{args.output_dir}'...")
        hf_dataset.save_to_disk(args.output_dir)
        hf_dataset.push_to_hub('sean-lamont/goedel_preprocessed', private=True)
        print("Done.")
    else:
        print("No samples passed the filters. No dataset saved.")


if __name__ == "__main__":
    # This check is crucial for multiprocessing on Windows/macOS
    main()