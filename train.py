import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM

from adapted_model import AdaptedEditFlowsTransformer
from datamodule import AdaptedDataModule
from adapt_qwen import create_adapted_qwen_model
from adapted_lightning import  AdaptedLitModule

def main():
    model_id = "Goedel-LM/Goedel-Prover-V2-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dm = AdaptedDataModule(tokenizer=tokenizer, batch_size=2)

    model = AdaptedEditFlowsTransformer(model_id)
    # hardcoded for now, since tokenizer length and model embedding matrix dimensions are different..
    lit_module = AdaptedLitModule(model, 151936, tokenizer.pad_token_id, 151651) #using <|quad_end|>

    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)
    trainer.fit(lit_module, dm)

if __name__ == "__main__":
    main()


# todo
# - Prep and load data from error correction runs
# - Add LoRA and quantization
# - Add/test generation and validation setup
