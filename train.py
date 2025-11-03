import lightning.pytorch as pl
import lightning.pytorch.profilers
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModelForCausalLM

from adapted_model import AdaptedEditFlowsTransformer
from datamodule import AdaptedDataModule
from adapted_lightning import  AdaptedLitModule
from lightning.pytorch.strategies import DeepSpeedStrategy
import pickle
import torch

from lightning.pytorch import Callback


profiler = lightning.pytorch.profilers.PyTorchProfiler(
    # Set the directory to save traces
    dirpath="tb_logs/profiler",

    # The file name for the trace
    filename="profile_trace",

    # --- These are the crucial options for memory ---

    # 1. Tell it to record memory events
    profile_memory=True,

    # 2. Tell it to record the Python stack trace
    with_stack=True,

    # 3. Tell it to record tensor shapes
    record_shapes=True
)





def main():
    model_id = "Goedel-LM/Goedel-Prover-V2-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dm = AdaptedDataModule(tokenizer=tokenizer, batch_size=1, full_vocab_size=151936)

    model = AdaptedEditFlowsTransformer(model_id)
    # hardcoded for now, since tokenizer length and model embedding matrix dimensions are different..
    lit_module = AdaptedLitModule(model, 151936, tokenizer.pad_token_id, 151651) #using <|quad_end|>



    wandb_logger = WandbLogger(project="edit-flows", name="standard strategy",  )
    # wandb_logger.watch(lit_module, log_freq=10, log='all')

    strategy = DeepSpeedStrategy(
    stage=2,
    offload_optimizer=True,
    offload_optimizer_device="cpu",
    # Other DeepSpeed parameters will use defaults unless specified
    exclude_frozen_parameters=True
    )

    # # update to checkpoint based on bleu score
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/bleu_score",
        mode="max",
        # dirpath="checkpoints",
        filename="best-checkpoint-{epoch:02d}-{val/bleu_score:.2f}",
        save_top_k=1,
        save_last=True
    )

    # mem_callback = MemorySnapshotCallback(8)
    #
    trainer = pl.Trainer(log_every_n_steps=8,
                         # strategy='deepspeed_stage_2_offload',
                         # strategy=strategy,
                         # profiler=profiler,
                         precision='bf16-mixed',
                         logger=wandb_logger,
                         # limit_train_batches=10,
                         limit_val_batches=200,
                         accumulate_grad_batches=32,
                         gradient_clip_val=1,
                         num_sanity_val_steps=0,
                         val_check_interval=0.2,
                         callbacks=[checkpoint_callback],#, mem_callback],
                         # detect_anomaly=True
                         )
    trainer.fit(lit_module, dm)




if __name__ == "__main__":
    main()
