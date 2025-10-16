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
    lit_module = AdaptedLitModule(model, len(tokenizer), tokenizer.pad_token_id, 151651) #using <|quad_end|>

    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)
    trainer.fit(lit_module, dm)

if __name__ == "__main__":
    main()
