# Edit Flows Implementation for Code Correction (PyTorch Lightning)

(Work in Progress)

This repository contains an unofficial PyTorch Lightning implementation of the main approach in the paper
[Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018) by Havasi et al.
The initial code was based on this repository: https://github.com/TheMatrixMaster/edit-flows-demo.

It extends the above implementation to focus on language modelling (targeting code correction),
with a more modular structure suitable for training and evaluating on different datasets and models.
It might be a useful starting point to experiment with the Edit Flows approach in other contexts.

It also allows adaptation of large pre-trained autoregressive models using a similar attention annealing approach from
the paper [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://arxiv.org/abs/2410.17891).

The primary example for the project adapts the [RepairLLAMA](https://github.com/ASSERT-KTH/repairllama) model and
dataset for code correction.

(It also includes code for a custom dataset for formal theorem proving based on Goedel-Prover-V2, which is a work in
progress).

## Status

**Note:** This repository is a work in progress.

The current implementation is functional for training and basic sampling, but further improvements and experiments are
planned.

- Currently only works for a batch size of 1 per GPU, with larger batch sizes as a todo.
- The sampling is basic and does not yet include the reverse rate model described in the paper.
- Only tested training for the fairly small RepairLLAMA model so far, and not fully to convergence.
Given this, the sample quality we observe is limited, but still shows signs of correctly learning to fix simple bugs.
This gives some confidence in the implementation. We've included some sample outputs in the `train_diff.html` and `val_diff.html` files
for sampling on the training and validation sets after a few training epochs.
- Planned experiments to compare adaptation against training from scratch. 


## Project Structure

The project is organized into several modules:

- **Data Handling**:
    - `datamodule.py`: A PyTorch Lightning `DataModule` to handle all data loading and preparation.
    - `dataset/`: Contains dataset-specific logic.
        - `HFDataset.py`: A PyTorch `Dataset` for loading data.
        - `preprocess_*.py`: Scripts for preprocessing the RepairLLAMA dataset and our custom Goedel-Prover-V2 based dataset.
    - `setup_tokenizer.py`: Utility for initializing the tokenizer to include addional GAP tokens.

- **Model Components**:
    - `adapted_model.py`: The core model definition, which combines a pre-trained transformer with the edit flow
      mechanism.
    - `couplings.py`: Implementation of the coupling class for source to target sequences.
    - `scheduler.py`: Defines the noise schedule for the flow process.

- **Training**:
    - `train.py`: The main script to execute the training process.
    - `adapted_lightning.py`: The PyTorch Lightning `LightningModule` that orchestrates the training, validation, and
      testing loops.

- **Utilities**:
    - `utils.py`: General helper functions.
    - `vis_diff.py`: A script for visualizing differences between model inputs, outputs and targets.

## Setup

1. Create and activate a conda environment from the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate edit-flows
   ```
2. Install torch through pip.
```bash
    pip install torch==2.8
```
3. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
4. Install flash-attn (if using compatible GPU):
    ```bash
    pip install flash_attn==2.8.3
    ```

## Usage

1. **Preprocess Data**: Preprocess data into the correct format, with RepairLLAMA as an example:
   ```bash
   python dataset/preprocess_repair_llama.py
   ```
   I provide a public HF dataset for RepairLLAMA code correction, so this step can be skipped if using that.
 
2. **Model Setup**: If using a different architecture, set up a model architecture using the format in `adapted_model.py` as a reference.
*IMPORTANT*: If adapting a pre-trained model with annealing, you might have to monkey-patch the base model to work with bidirectional attention.
For RepairLLAMA/CodeLlama, this should be in the `forward` function from `modeling_llama` file, to ignore `_prepare_decoder_attention_mask`.
 
3. **Configure Training**: Modify the hyperparameters and settings in `train.py` as needed. Changes to the training and sampling logic can be made in `adapted_lightning.py`.

4. **Train Model**: Start the training process by running the `train.py` script.
   ```bash
   python train.py
   ```

## TODO

- [ ] Implement support for larger batch sizes per GPU.
- [ ] Implement and train the reverse rate model described in the paper for more accurate sampling.
- [ ] Experiments to compare the performance of model adaptation against training a similar model from scratch.
