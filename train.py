import lightning.pytorch as pl
from datamodule import GoedelDataModule
from adapt_qwen import create_adapted_qwen_model
from qwen_lightning import QwenLitModule

def main():
    model_id = "Goedel-LM/Goedel-Prover-V2-8B"
    model, tokenizer = create_adapted_qwen_model(model_id)

    # Example data from test_goedel.ipynb
    formal_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat


theorem square_equation_solution {x y : ℝ} (h : x^2 + y^2 = 2*x - 4*y - 5) : x + y = -1 := by
  sorry
"""
    data = [
        {'context': formal_statement, 'response': '  rw [← sub_eq_zero] at h\n  have h_rw : (x - 1)^2 + (y + 2)^2 = 0 := by\n    linarith\n  rw [add_eq_zero_iff_eq_zero_and_eq_zero] at h_rw\n  rcases h_rw with ⟨h_x, h_y⟩\n  have h_x_1 : x = 1 := by\n    rwa [pow_two, mul_self_eq_zero] at h_x\n  have h_y_1 : y = -2 := by\n    rwa [pow_two, mul_self_eq_zero] at h_y\n  simp [h_x_1, h_y_1]'}, 
    ]

    dm = GoedelDataModule(tokenizer=tokenizer, batch_size=1, data=data)

    lit_module = QwenLitModule(model, lr=1e-5, anneal_total_steps=1000)

    trainer = pl.Trainer(max_epochs=1, log_every_n_steps=1)
    trainer.fit(lit_module, dm)

if __name__ == "__main__":
    main()
