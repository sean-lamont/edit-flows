import lightning.pytorch as pl
from datamodule import AdaptedDataModule
from adapt_qwen import create_adapted_qwen_model
from adapted_lightning import  AdaptedLitModule

def main():
    model_id = "Goedel-LM/Goedel-Prover-V2-8B"
    model, tokenizer = create_adapted_qwen_model(model_id)

    dm = AdaptedDataModule(tokenizer=tokenizer, batch_size=2)

    lit_module = AdaptedLitModule(model, len(tokenizer), tokenizer.pad_token_id, 151651) #using <|quad_end|>

    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)
    trainer.fit(lit_module, dm)

if __name__ == "__main__":
    main()
