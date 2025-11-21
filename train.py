"""
This is the main training script for the Edit Flows project.

It initializes the DataModule, the adapted model within the LightningModule, and a
PyTorch Lightning Trainer. It then starts the training and validation process.
Configuration for the training run can be modified within this script.
"""


import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
from transformers import AutoTokenizer

from adapted_model import AdaptedEditFlowsTransformer
from datamodule import AdaptedDataModule
from adapted_lightning import  AdaptedLitModule
from models.sinusoidal_transformer import SimpleEditFlowsTransformer
import torch

from setup_tokenizer import get_model_and_tokenizer_info


def main():
    # Change this to 'sinusoidal' to run the sinusoidal experiment
    run_config = "hf"  # or "sinusoidal"

    if run_config == "sinusoidal":
        vocab_size = 128
        pad_token_id = 129
        bos_token_id = 128
        gap_token_id = 130
        full_vocab_size = 131

        class DummyTokenizer:
            def __init__(self):
                self.pad_token_id = pad_token_id
            def decode(self, token_ids, skip_special_tokens=False):
                return str(token_ids)

        tokenizer = DummyTokenizer()

        dm = AdaptedDataModule(dataset='sinusoidal', tokenizer=tokenizer, batch_size=128,
                               full_vocab_size=full_vocab_size, gap_token=gap_token_id,
                               dataset_cfg={'seq_len': 128, 'vocab_size': vocab_size})

        model = SimpleEditFlowsTransformer(
            vocab_size=full_vocab_size,
            hidden_dim=512,
            num_layers=8,
            num_heads=32,
            max_seq_len=256,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
        )
        lit_module = AdaptedLitModule(model, tokenizer, pad_token_id, gap_token_id)
        wandb_logger = WandbLogger(project="edit-flows", name="sinusoidal_test")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            filename="best-checkpoint-{epoch:02d}-{val/loss:.2f}",
            save_top_k=1,
            save_last=True
        )
        trainer = pl.Trainer(log_every_n_steps=8,
                             precision='bf16-mixed',
                             logger=wandb_logger,
                             limit_val_batches=10,
                             accumulate_grad_batches=1,
                             gradient_clip_val=1,
                             num_sanity_val_steps=0,
                             val_check_interval=0.2,
                             callbacks=[checkpoint_callback],
                             )

    else:
        model_id = "TheBloke/CodeLlama-7B-fp16"
        lora_id = "ASSERT-KTH/RepairLLaMA-IR1-OR1"

        tokenizer, gap_token_id, full_vocab_size = get_model_and_tokenizer_info(
            base_model_id=model_id,
            lora_adapter_id=lora_id,
            special_token="<GAP>",
            torch_dtype=torch.float16,
        )

        dm = AdaptedDataModule(dataset='sean-lamont/repairllama_preprocessed', tokenizer=tokenizer, batch_size=1,
                               full_vocab_size=full_vocab_size, gap_token=gap_token_id)

        model = AdaptedEditFlowsTransformer(model_id, lora_id=lora_id)

        lit_module = AdaptedLitModule(model, tokenizer, tokenizer.pad_token_id, gap_token_id)

        wandb_logger = WandbLogger(project="code-repair", name="train_first_ckpt",  offline=False,
                                   group = "Val Compare 1000")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val/bleu_score",
            mode="max",
            filename="best-checkpoint-{epoch:02d}-{val/bleu_score:.2f}",
            save_top_k=1,
            save_last=True
        )

        trainer = pl.Trainer(log_every_n_steps=8,
                             precision='bf16-mixed',
                             logger=wandb_logger,
                             limit_val_batches=1000,
                             accumulate_grad_batches=32,
                             gradient_clip_val=1,
                             num_sanity_val_steps=0,
                             val_check_interval=0.2,
                             callbacks=[checkpoint_callback],
                             )

    trainer.fit(lit_module, dm)


if __name__ == "__main__":
    main()
