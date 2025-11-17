"""
This is the main training script for the Edit Flows project.

It initializes the DataModule, the adapted model within the LightningModule, and a
PyTorch Lightning Trainer. It then starts the training and validation process.
Configuration for the training run can be modified within this script.
"""


import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

from adapted_model import AdaptedEditFlowsTransformer
from datamodule import AdaptedDataModule
from adapted_lightning import  AdaptedLitModule
from lightning.pytorch.strategies import DeepSpeedStrategy
import torch

from lightning.pytorch import Callback

from dataset.HFDataset import HFDataset
from setup_tokenizer import get_model_and_tokenizer_info


def main():
    model_id = "TheBloke/CodeLlama-7B-fp16"
    lora_id = "ASSERT-KTH/RepairLLaMA-IR1-OR1" # need to overwrite modeling_llama file to ignore _prepare_decoder_attention_mask

    # adds token to tokenizer if it doesn't exist, gets max of tokenizer length and model emb matrix
    tokenizer, gap_token_id, full_vocab_size = get_model_and_tokenizer_info(
        base_model_id="TheBloke/CodeLlama-7B-fp16",
        lora_adapter_id="ASSERT-KTH/RepairLLaMA-IR1-OR1",
        special_token="<GAP>",
        torch_dtype=torch.float16,
    )

    # ds = HFDataset('sean-lamont/repairllama_preprocessed')  # 'sean-lamont/goedel_preprocessed'
    dm = AdaptedDataModule(dataset='sean-lamont/repairllama_preprocessed', tokenizer=tokenizer, batch_size=1,
                           full_vocab_size=full_vocab_size, gap_token=gap_token_id)

    model = AdaptedEditFlowsTransformer(model_id, lora_id=lora_id)

    lit_module = AdaptedLitModule(model, tokenizer, tokenizer.pad_token_id, gap_token_id)


    wandb_logger = WandbLogger(project="code-repair", name="train_first_ckpt",  offline=False,
                               group = "Val Compare 1000")

    # update to checkpoint based on bleu score
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
                         # detect_anomaly=True
                         )

    trainer.fit(lit_module, dm)




    # dm.setup('fit')
    #
    # # trainer.validate(lit_module, dm, ckpt_path = 'code-repair/7shrv5ml/checkpoints/last.ckpt')
    # # trainer.validate(lit_module, dm, ckpt_path = 'code-repair/7shrv5ml/checkpoints/last.ckpt')
    # # trainer.validate(lit_module, dm.train_dataloader(), ckpt_path = 'code-repair/7shrv5ml/checkpoints/last.ckpt')
    #
    # trainer.validate(lit_module, dm.train_dataloader(), ckpt_path = 'code-repair/7shrv5ml/checkpoints/best-checkpoint-epoch=11-val/bleu_score=97.40.ckpt')
    # wandb.finish()
    #
    # wandb_logger=WandbLogger(project="code-repair", name="val_first_ckpt", offline=False,group="Val Compare 1000" )
    # trainer.logger = wandb_logger
    # trainer.validate(lit_module, dm, ckpt_path = 'code-repair/7shrv5ml/checkpoints/best-checkpoint-epoch=11-val/bleu_score=97.40.ckpt')
    # wandb.finish()
    #
    # wandb_logger=WandbLogger(project="code-repair", name="train_second_ckpt", offline=False,group="Val Compare 1000" )
    # trainer.logger = wandb_logger
    # trainer.validate(lit_module, dm.train_dataloader(), ckpt_path = 'code-repair/7shrv5ml/checkpoints/last.ckpt')
    # wandb.finish()
    #
    # wandb_logger=WandbLogger(project="code-repair", name="val_second_ckpt", offline=False, group="Val Compare 1000" )
    # trainer.logger = wandb_logger
    # trainer.validate(lit_module, dm, ckpt_path = 'code-repair/7shrv5ml/checkpoints/last.ckpt')
    # wandb.finish()


if __name__ == "__main__":
    main()
