import datasets
import torch
import transformers
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, tokenizer, mode='llada', max_length=1024, gap_token_id=3):
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.gap_token_id = gap_token_id

        # LLaDA uses Left Padding
        self.tokenizer.padding_side = 'left'

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _apply_chat_template(self, prompt):
        return LLADA_TEMPLATE.format(prompt=prompt)

    def __call__(self, batch):
        prompts = [self._apply_chat_template(b['prompt']) for b in batch]
        codes = [b['code'] + self.tokenizer.eos_token for b in batch]

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

    def _collate_tef(self, prompts, codes):
        z0_list, z1_list = [], []
        context_mask_list = []

        for p_str, c_str in zip(prompts, codes):
            p_ids = self.tokenizer.encode(p_str, add_special_tokens=False)
            c_ids = self.tokenizer.encode(c_str, add_special_tokens=False)

            # Target: Prompt + Code
            z1_seq = p_ids + c_ids

            # Source: Prompt + Gaps (Insertion Task)
            z0_seq = p_ids + [self.gap_token_id] * len(c_ids)

            c_mask = [1] * len(p_ids) + [0] * len(c_ids)

            if len(z1_seq) > self.max_length:
                z1_seq = z1_seq[:self.max_length]
                z0_seq = z0_seq[:self.max_length]
                c_mask = c_mask[:self.max_length]

            z0_list.append(torch.tensor(z0_seq, dtype=torch.long))
            z1_list.append(torch.tensor(z1_seq, dtype=torch.long))
            context_mask_list.append(torch.tensor(c_mask, dtype=torch.bool))

        z0 = self._left_pad(z0_list, self.tokenizer.pad_token_id)
        z1 = self._left_pad(z1_list, self.tokenizer.pad_token_id)
        context_mask = self._left_pad(context_mask_list, 1)

        x_1 = z1.clone()
        x_0 = z0.clone()
        x_0[x_0 == self.gap_token_id] = self.tokenizer.pad_token_id

        t = torch.rand(z0.shape[0], 1)

        return x_0, x_1, z0, z1, t, context_mask

    def _left_pad(self, tensor_list, padding_value):
        reversed_tensors = [t.flip(0) for t in tensor_list]
        padded = torch.nn.utils.rnn.pad_sequence(
            reversed_tensors, batch_first=True, padding_value=padding_value
        )
        return padded.flip(1)


def get_dataloaders(tokenizer, batch_size=4, mode='llada', max_length=1024):
    train_ds = AutoformalizationDataset('train', tokenizer, max_length)
    val_ds = AutoformalizationDataset('test', tokenizer, max_length)

    collator = AutoformalCollator(tokenizer, mode=mode, max_length=max_length)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collator, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=4
    )
    return train_loader, val_loader

#
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
#     is_gap = (z0 == 999)
#     print(f"Number of Gaps inserted: {is_gap.sum().item()}")
#     print("Test Complete.")
