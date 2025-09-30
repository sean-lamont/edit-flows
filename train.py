import lightning.pytorch as pl
from datamodule import SinusoidDataModule
from simpleeditflowtransformer import SimpleEditFlowsTransformer
from editflow_lightning import EditFlowLitModule  # adjust import path if needed
from utils import *

def main():
    model_core = SimpleEditFlowsTransformer(vocab_size=BASE_VOCAB + 2, hidden_dim=256,
                                            num_layers=4, num_heads=32, max_seq_len=256,)
    lit = EditFlowLitModule(model_core, lr=1e-4, scheduler_cfg={'a':1.0,'b':1.0})
    dm = SinusoidDataModule(batch_size=128, n_train=8000, n_val=512)
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=50)
    trainer.fit(lit, dm)

if __name__ == "__main__":
    main()