import numpy as np
import torch
from pathlib import Path

from constants import BOS_TOKEN, PAD_TOKEN
from model import SimpleEditFlowsTransformer
from utils import load_model_state
from training import train_model
from sampling import run_sampling
from config import V, L

# torch.manual_seed(42)
# np.random.seed(42)

model = SimpleEditFlowsTransformer(
    vocab_size=V + 2,  # +2 for PAD + BOS tokens
    hidden_dim=512,
    num_layers=8,
    num_heads=32,
    max_seq_len=2 * L,
    pad_token_id=PAD_TOKEN,
    bos_token_id=BOS_TOKEN,
)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
# optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
#
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Call the training function

# (Optional) Save / Load the model state
train = True
overwrite = True

save_dir = Path(f"results/empty_coupling")
save_dir.mkdir(parents=True, exist_ok=True)
model_name = Path(f"seq2seq_prior.pt")

if train:
    model, optim = train_model(model, optim, device, V)
    save_path = save_dir / model_name
    if not overwrite:
        assert not save_path.exists(), f"Model file {save_path} already exists. Please choose a different name."
    assert save_path.parent.exists(), f"Directory {save_path.parent} does not exist. Please create it first."
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'vocab_size': model.vocab_size,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'num_heads': model.num_heads,
        'max_seq_len': model.max_seq_len,
        'bos_token_id': model.bos_token_id,
        'pad_token_id': model.pad_token_id,
    }, save_path)
    print(f"Model saved to {save_path}")

    # Run sampling and visualization
    run_sampling(model, device, V)

else:
    save_path = 'best_model.pt'
    checkpoint = torch.load(save_path)#, map_location={'cpu': 'cuda'})
    model.load_state_dict(checkpoint)
    model.to(device)
    print(f"Model loaded from {save_path}")
    # Run sampling and visualization
    run_sampling(model, device, V)


