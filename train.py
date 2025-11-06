import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModelForCausalLM

from adapted_model import AdaptedEditFlowsTransformer
from datamodule import AdaptedDataModule
from adapted_lightning import  AdaptedLitModule
from lightning.pytorch.strategies import DeepSpeedStrategy
import torch

from lightning.pytorch import Callback

from dataset.goedel_dataset import GoedelDataset


def main():
    FULL_VOCAB_SIZE = 151936
    GAP_TOKEN_ID = 151651

    model_id = "Goedel-LM/Goedel-Prover-V2-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ds = GoedelDataset()

    dm = AdaptedDataModule(dataset=ds, tokenizer=tokenizer, batch_size=1, full_vocab_size=FULL_VOCAB_SIZE)

    model = AdaptedEditFlowsTransformer(model_id)

    lit_module = AdaptedLitModule(model, FULL_VOCAB_SIZE, tokenizer, tokenizer.pad_token_id, GAP_TOKEN_ID) #using <|quad_end|> for Goedel


    wandb_logger = WandbLogger(project="edit-flows", name="correction_only",  )

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
                         limit_val_batches=200,
                         accumulate_grad_batches=32,
                         gradient_clip_val=1,
                         num_sanity_val_steps=0,
                         val_check_interval=0.2,
                         callbacks=[checkpoint_callback],
                         )

    trainer.fit(lit_module, dm)

if __name__ == "__main__":
    main()
