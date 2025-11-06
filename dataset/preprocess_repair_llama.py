import datasets
from datasets import load_dataset

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

from setup_tokenizer import get_model_and_tokenizer_info

# CRITICAL: Assumes 'utils.py' is in the same directory or Python path
# so that worker processes can import it.
try:
    from utils import opt_align_xs_to_zs
except ImportError:
    print("Error: Could not import 'opt_align_xs_to_zs' from 'utils.py'.")
    print("Please make sure 'preprocess.py' is in the same directory as 'utils.py'.")
    exit(1)

# --- obtain from setup_tokenizer --

# --- Global tokenizer for worker processes ---
# This will be initialized once per worker in init_worker


def process_dataset(args: argparse.Namespace):
    tokenizer, gap_token_id, full_vocab = get_model_and_tokenizer_info(
        base_model_id="TheBloke/CodeLlama-7B-fp16",
        lora_adapter_id="ASSERT-KTH/RepairLLaMA-IR1-OR1",
        special_token="<GAP>",
        torch_dtype=torch.float16
    )

    PAD_TOKEN_ID = tokenizer.pad_token_id

    def _process_sample(sample):
        context = 'CONTEXT' # set as something nonzero
        prev_attempt = sample['input']
        target = sample['output']

        context_ids = tokenizer(context, return_tensors='pt').input_ids.squeeze(0)
        prev_ids = tokenizer(prev_attempt, return_tensors='pt').input_ids.squeeze(0)
        target_ids = tokenizer(target, return_tensors='pt').input_ids.squeeze(0)

        # # --- Check Original Length Filters (x0, x1) ---
        # if prev_ids.shape[0] > args.max_len or target_ids.shape[0] > args.max_len:
        #     return None

        # --- Truncate context (from datamodule.py) ---
        context_ids = context_ids[:args.max_context_len]

        # --- Run Alignment (from collate.py) ---
        # This is the most expensive CPU step
        z0, z1 = opt_align_xs_to_zs(
            prev_ids.unsqueeze(0),
            target_ids.unsqueeze(0),
            PAD_TOKEN_ID,
            gap_token_id,
        )
        z0 = z0.squeeze(0)
        z1 = z1.squeeze(0)

        # --- Check Aligned Length Filter (z) ---
        # if z0.shape[0] > args.max_z_len:
        #     return None

        # --- This sample is "good". Add its data to the list. ---
        output_data = {
            'x0': prev_ids.long().tolist(),
            'z0': z0.long().tolist(),
            'z1': z1.long().tolist(),
            'x1': target_ids.long().tolist(),
            'context': context_ids.long().tolist(),
            'type': 'correction'
        }
        return output_data

    # Load ir1xor1 (full function to full function)
    dataset = load_dataset("ASSERT-KTH/repairllama-datasets", "ir1xor1")
    print (dataset)

    num_processors = os.cpu_count()

    # save as hf dataset, keeping train and val as they are split
    processed_dataset = dataset.map(_process_sample, num_proc=num_processors,
                                    remove_columns=dataset['train'].column_names)



    # Filter out None values (samples that didn't pass the filters)
    processed_dataset = processed_dataset.filter(lambda x: x is not None)

    # filter out values where x0 + x1 > 1024
    processed_dataset = processed_dataset.filter(lambda x: len(x['x0']) + len(x['x1']) <= 1024)

    # save to hf hub
    processed_dataset.push_to_hub('sean-lamont/repairllama_preprocessed', private=True)

    return


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-process RepairLLAMA dataset for edit flows.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="ASSERT-KTH/RepairLLaMA-IR1-OR1",
        help="Name of the Hugging Face tokenizer (e.g., 'gpt2', 'bert-base-uncased')."
    )

    # --- Filtering Thresholds ---
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=1024,
        help="Max token length for context. Samples longer than this are truncated."
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1024,
        help="Max token length for prev_attempt (x0) and target (x1). Samples longer are FILTERED."
    )
    parser.add_argument(
        "--max_z_len",
        type=int,
        default=2048,
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


if __name__ == "__main__":
    # This check is crucial for multiprocessing on Windows/macOS
    args = parse_args()
    process_dataset(args)
