import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModelForCausalLM

from adapted_model import AdaptedEditFlowsTransformer
from datamodule import AdaptedDataModule
from adapted_lightning import  AdaptedLitModule
from lightning.pytorch.strategies import DeepSpeedStrategy

def main():
    model_id = "Goedel-LM/Goedel-Prover-V2-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dm = AdaptedDataModule(tokenizer=tokenizer, batch_size=1)

    model = AdaptedEditFlowsTransformer(model_id)
    # hardcoded for now, since tokenizer length and model embedding matrix dimensions are different..
    lit_module = AdaptedLitModule(model, 151936, tokenizer.pad_token_id, 151651) #using <|quad_end|>


    wandb_logger = WandbLogger(project="edit-flows", name="test_1",  )
    # wandb_logger.watch(lit_module, log_freq=10)

    strategy = DeepSpeedStrategy(
    stage=2,
    offload_optimizer=True,
    offload_optimizer_device="cpu",
    # Other DeepSpeed parameters will use defaults unless specified
    exclude_frozen_parameters=True
    )

    # update to checkpoint based on bleu score
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_bleu_score",
        mode="max",
        # dirpath="checkpoints",
        filename="best-checkpoint-{epoch:02d}-{val_bleu_score:.2f}",
        save_top_k=1,
        save_last=True
    )

    trainer = pl.Trainer(log_every_n_steps=1,
                         # strategy='deepspeed_stage_2_offload',
                         strategy=strategy,
                         precision='bf16-mixed',
                         logger=wandb_logger,
                         accumulate_grad_batches=8,
                         gradient_clip_val=5,
                         # num_sanity_val_steps=1,
                         val_check_interval=0.25,
                         callbacks=[checkpoint_callback],
                         )
    trainer.fit(lit_module, dm)

if __name__ == "__main__":
    main()
