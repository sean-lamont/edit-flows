import datasets
import torch
import transformers
import tokenizers
from torch.utils.data import DataLoader, Dataset
from dataloader import get_tokenizer

LLADA_TEMPLATE = (
    "<BOS><start_id>user<end_id>\n{prompt}<eot_id>"
    "<start_id>assistant<end_id>\n"
)


class AutoformalizationDataset(Dataset):
    def __init__(self, split='train', tokenizer=None, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Loading Autoformalization dataset ({split})...")
        self.dataset = datasets.load_dataset(
            "casey-martin/multilingual-mathematical-autoformalization",
            "lean",
            split=split,
            trust_remote_code=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "prompt": item["input"],
            "code": item["output"]
        }


class AutoformalCollator:
    def __init__(self, tokenizer, mode='llada', max_length=1024, gap_token_id=126084, pad_token_id=126085):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.gap_token_id = gap_token_id

        # LLaDA uses Left Padding
        self.tokenizer.padding_side = 'right'

        # for llada, set padding token to something else than eos_token_id to avoid issues from rainbow padding paper

        self.tokenizer.pad_token_id = pad_token_id
        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.min_code_len = self.max_length // 2

    def _apply_chat_template(self, prompt):
        return LLADA_TEMPLATE.format(prompt=prompt)

    def __call__(self, batch):
        prompts = [self._apply_chat_template(b['prompt']) for b in batch]
        # codes = [b['code'] + self.tokenizer.eos_token for b in batch]

        # do not add eos, let the model learn this naturally
        codes = [b['code'] for b in batch]

        if self.mode == 'llada':
            return self._collate_llada(prompts, codes)
        elif self.mode == 'tef':
            return self._collate_tef(prompts, codes)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _collate_llada(self, prompts, codes):
        input_ids_list = []
        context_masks_list = []

        for p_str, c_str in zip(prompts, codes):
            p_ids = self.tokenizer.encode(p_str, add_special_tokens=False)
            c_ids = self.tokenizer.encode(c_str, add_special_tokens=False)

            full_ids = p_ids + c_ids
            # Mask: 1 for prompt (frozen), 0 for code (train)
            mask = [1] * len(p_ids) + [0] * len(c_ids)

            if len(full_ids) > self.max_length:
                full_ids = full_ids[:self.max_length]
                mask = mask[:self.max_length]

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
            context_masks_list.append(torch.tensor(mask, dtype=torch.bool))

        input_ids = self._left_pad(input_ids_list, self.tokenizer.pad_token_id)
        context_masks = self._left_pad(context_masks_list, 1)  # Pad context with 1 (ignore)

        return {
            "input_ids": input_ids,
            "context_mask": context_masks
        }

    def _collate_tef(self, prompts, codes, del_prob=0.05):
        # Clean endpoints (different lengths)
        x0_list, x1_list = [], []

        # Aligned training trajectories (same length)
        z0_list, z1_list = [], []
        context_mask_list = []

        for p_str, c_str in zip(prompts, codes):
            p_ids = self.tokenizer.encode(p_str, add_special_tokens=False)
            c_ids = self.tokenizer.encode(c_str, add_special_tokens=False)

            if p_ids.shape[0] + c_ids.shape[0] > self.max_length:
                # ensure at least min(c_ids.shape[0], min_code_len) of code is preserved
                code_len = min(c_ids.shape[0], self.min_code_len)
                p_ids = p_ids[:(self.max_length - code_len)]
                c_ids = c_ids[:code_len]

            curr_x0 = p_ids
            curr_x1 = p_ids + c_ids

            x0_list.append(torch.tensor(curr_x0, dtype=torch.long))
            x1_list.append(torch.tensor(curr_x1, dtype=torch.long))

            prefix = p_ids

            n_code_tokens = len(c_ids)
            n_slots = n_code_tokens  # +1

            # Noise Logic: Only applied to the Code region
            ins_mask = torch.rand(n_slots) < del_prob

            z0_code_parts = []
            z1_code_parts = []

            if ins_mask.any():
                num_ins = ins_mask.sum().item()

                # Generate Garbage tokens (Sample from vocab, reject Pad/Gap/BOS)
                noise_tokens = torch.randint(0, self.tokenizer.vocab_size, (num_ins,))
                mask_invalid = (noise_tokens == self.tokenizer.pad_token_id) | \
                               (noise_tokens == self.gap_token_id) | \
                               (noise_tokens == self.tokenizer.bos_token_id)

                while mask_invalid.any():
                    num_invalid = mask_invalid.sum().item()
                    new_tokens = torch.randint(0, self.tokenizer.vocab_size, (num_invalid,))
                    noise_tokens[mask_invalid] = new_tokens
                    mask_invalid = (noise_tokens == self.tokenizer.pad_token_id) | \
                                   (noise_tokens == self.gap_token_id) | \
                                   (noise_tokens == self.tokenizer.bos_token_id)

                noise_tokens = noise_tokens.tolist()
                noise_ptr = 0

                for i in range(n_slots):
                    # Insertion Logic (Garbage in z0, Gap in z1)
                    if ins_mask[i]:
                        z0_code_parts.append(noise_tokens[noise_ptr])
                        z1_code_parts.append(self.gap_token_id)
                        noise_ptr += 1

                    # Standard Token Logic (Gap in z0, Code in z1)
                    # if i < n_code_tokens:
                    else:
                        z0_code_parts.append(self.gap_token_id)
                        z1_code_parts.append(c_ids[i])
            else:
                # No noise path
                z0_code_parts = [self.gap_token_id] * n_code_tokens
                z1_code_parts = c_ids

            # Combine Prefix + Mutable Region
            z0_seq = prefix + z0_code_parts
            z1_seq = prefix + z1_code_parts

            # Mask: 1 for Immutable Prefix, 0 for Mutable Region (Code + Noise)
            c_mask = [1] * len(prefix) + [0] * len(z0_code_parts)

            # # Truncation (applied to the aligned sequences)
            # if len(z1_seq) > self.max_length:
            #     z1_seq = z1_seq[:self.max_length]
            #     z0_seq = z0_seq[:self.max_length]
            #     c_mask = c_mask[:self.max_length]

            z0_list.append(torch.tensor(z0_seq, dtype=torch.long))
            z1_list.append(torch.tensor(z1_seq, dtype=torch.long))
            context_mask_list.append(torch.tensor(c_mask, dtype=torch.bool))

        # Pad each list independently to its own max length.
        # x_0 = self._left_pad(x0_list, self.tokenizer.pad_token_id)
        # x_1 = self._left_pad(x1_list, self.tokenizer.pad_token_id)
        #
        # z0 = self._left_pad(z0_list, self.tokenizer.pad_token_id)
        # z1 = self._left_pad(z1_list, self.tokenizer.pad_token_id)

        x_0 = self._right_pad(x0_list, self.tokenizer.pad_token_id)
        x_1 = self._right_pad(x1_list, self.tokenizer.pad_token_id)
        z0 = self._right_pad(z0_list, self.tokenizer.pad_token_id)
        z1 = self._right_pad(z1_list, self.tokenizer.pad_token_id)

        context_mask = self._right_pad(context_mask_list, 0)

        t = torch.rand(x_1.shape[0], 1)
        t = torch.clamp(t, min=0.01, max=0.99)

        return x_0, x_1, z0, z1, t, context_mask

    # dataloader_af.py

    # Rename/Modify this function
    def _right_pad(self, tensor_list, padding_value):
        # Standard pad_sequence is Right Padding by default for batch_first=True
        return torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=padding_value
        )

    def _left_pad(self, tensor_list, padding_value):
        reversed_tensors = [t.flip(0) for t in tensor_list]
        padded = torch.nn.utils.rnn.pad_sequence(
            reversed_tensors, batch_first=True, padding_value=padding_value
        )
        return padded.flip(1)


def get_dataloaders(tokenizer, batch_size=4, mode='llada', max_length=1024):
    if tokenizer.padding_side != 'right':
        tokenizer.padding_side = 'right'

    train_ds = AutoformalizationDataset('train', tokenizer, max_length)
    val_ds = AutoformalizationDataset('test', tokenizer, max_length)

    collator = AutoformalCollator(tokenizer, mode=mode, max_length=max_length, gap_token_id=126084)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=4
    )

    train_loader.tokenizer = tokenizer
    val_loader.tokenizer = tokenizer

    return train_loader, val_loader

# if __name__ == "__main__":
#     print("Initializing Tokenizer...")
#     tokenizer = transformers.AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
#
#     print("\n--- Testing LLaDA Mode (Left Padding) ---")
#     ds = AutoformalizationDataset(split='train', tokenizer=tokenizer)
#     collator = AutoformalCollator(tokenizer, mode='llada', max_length=128, gap_token_id=126084)
#
#     # Create a mock batch
#     mock_batch = [ds[0], ds[1]]
#     print(f"Sample 0 Input: {mock_batch[0]['prompt'][:50]}...")
#     print(f"Sample 0 Output: {mock_batch[0]['code'][:50]}...")
#
#     batch_out = collator(mock_batch)
#     inp = batch_out['input_ids']
#     mask = batch_out['context_mask']
#
#     print(f"Batch Shape: {inp.shape}")
#     print(f"Left Padding Check (First token should be pad if lengths differ): {inp[0, 0]}")
#     print(f"Decoded Sample 0 (with padding):")
#     print(tokenizer.decode(inp[0]))
#
#     print("\n--- Testing TEF Mode (Insertion Alignment) ---")
#     collator_tef = AutoformalCollator(tokenizer, mode='tef', max_length=128, gap_token_id=126084)
#     x0, x1, z0, z1, t, ctx_mask = collator_tef(mock_batch)
#
#     print(f"z0 Shape: {z0.shape}")
#     print(f"z1 Shape: {z1.shape}")
#
#     print(f"z0 last tokens (sample 0): {z0[0, -10:].tolist()}")
#     print(f"z1 last tokens (sample 0): {z1[0, -10:].tolist()}")
#
#     print(f"decoded z0 (sample 0): {tokenizer.decode(z0[0])}\n\n")
#     print(f"decoded z1 (sample 0): {tokenizer.decode(z1[0])}\n\n")
#
#
#
#     is_gap = (z0 == 126084)
#     print(f"Number of Gaps inserted: {is_gap.sum().item()}")
#     print("Test Complete.")
