import lightning.pytorch as pl
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

    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1,
                         strategy='deepspeed_stage_3_offload',
                         precision='bf16-mixed',
                         gradient_clip_val=1.0,
                         num_sanity_val_steps=0)
    trainer.fit(lit_module, dm)

if __name__ == "__main__":
    main()
