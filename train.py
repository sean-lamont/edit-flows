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
from setup_tokenizer import get_model_and_tokenizer_info


def main():
    # FULL_VOCAB_SIZE = 151936
    # GAP_TOKEN_ID = 151651
    # GAP_TOKEN = '<|quad_end|>'
    GAP_TOKEN = '<GAP>'

    # model_id = "Goedel-LM/Goedel-Prover-V2-8B"


    # tokenizer, gap_token_id, full_vocab_size = get_model_and_tokenizer_info(model_id, special_token=GAP_TOKEN)

    model_id = "TheBloke/CodeLlama-7B-fp16"
    lora_id = "ASSERT-KTH/RepairLLaMA-IR1-OR1"

    tokenizer, gap_token_id, full_vocab_size = get_model_and_tokenizer_info(
        base_model_id="TheBloke/CodeLlama-7B-fp16",
        lora_adapter_id="ASSERT-KTH/RepairLLaMA-IR1-OR1",
        special_token="<GAP>",
        torch_dtype=torch.float16
    )

    ds = GoedelDataset()

    dm = AdaptedDataModule(dataset=ds, tokenizer=tokenizer, batch_size=1, full_vocab_size=full_vocab_size)

    model = AdaptedEditFlowsTransformer(model_id, lora_id=lora_id)

    lit_module = AdaptedLitModule(model, full_vocab_size, tokenizer, tokenizer.pad_token_id, gap_token_id) #using <|quad_end|> for Goedel


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
