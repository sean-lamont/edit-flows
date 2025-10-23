import lightning.pytorch as pl
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from torch.nn import functional as F
from torch.optim import Optimizer
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from scheduler import CubicScheduler
from utils import *
from lightning.pytorch.utilities import grad_norm
import torchviz



class AdaptedLitModule(pl.LightningModule):
    def __init__(self, model: nn.Module, full_vocab_size, pad_token_id, gap_token_id, lr=5e-5, scheduler_cfg=None,
                 anneal_end_step=10000):
        super().__init__()
        # self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.kappa = CubicScheduler(**(scheduler_cfg or {'a': 1.0, 'b': 1.0}))
        self.anneal_end_step = anneal_end_step
        self.full_vocab_size = full_vocab_size
        self.pad_token = pad_token_id
        self.gap_token = gap_token_id
        self.lr = lr
        self.tokenizer = AutoTokenizer.from_pretrained("Goedel-LM/Goedel-Prover-V2-8B")
        self._oom_in_backward = False

    # def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # for name,param in self.named_parameters():
        #     if param.grad is not None:
        #         print(f"Layer: {name}, Gradient Norm: {param.grad.norm().item():.4f}")
            # else:
            #     print(f"Layer: {name}, No gradient")

        # print last lr from scheduler
        # print (f'scheduler lr: {self.lr_schedulers().get_last_lr()}')

    def configure_optimizers(self):
        # print (f'lr: {self.lr}')
        return DeepSpeedCPUAdam(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        # return DeepSpeedCPUAdam(self.parameters(), 1e-5, eps=1e-6)

        return torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95))

        # opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        #
        # steps = 500000 # change based on total steps
        #
        # scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=2000, num_training_steps=steps)
        #
        # return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}


    def forward(self, tokens, t, pad_mask, attn_mask_ratio):
        return self.model(tokens, t, pad_mask, attn_mask_ratio)

    def _loss(self, batch):
        x1, x0, z0, z1, t, context_lens = batch['x1'], batch['x0'], batch['z0'], batch['z1'], batch['t'], batch[
            'context_lens']

        p0 = x2prob(z0, self.full_vocab_size)
        p1 = x2prob(z1, self.full_vocab_size)

        zt = sample_cond_pt(p0, p1, t, self.kappa)

        xt, x_pad, z_gap, z_pad = rm_gap_tokens(zt, self.pad_token, self.gap_token)

        attn_mask_ratio = min(1.0, self.global_step / self.anneal_end_step)

        rates, ins_probs, sub_probs = self(xt, t, x_pad, attn_mask_ratio)

        # check that xt is the same as zt up to the context lengths for each sample:
        # for i in range(len(context_lens)):
            # assert torch.equal(xt[i, :context_lens[i]], x1[i, :context_lens[i]])
            # assert torch.equal(xt[i, :context_lens[i]], zt[i, :context_lens[i]])


            # print('x1\n\n')
            # print (self.tokenizer.batch_decode(x1))
            # print('xt\n\n')
            # print (self.tokenizer.batch_decode(xt))
            # print('\n\nz1\n\n')
            # print (self.tokenizer.batch_decode(z1))
            # print('\n\nzt\n\n')
            # print (self.tokenizer.batch_decode(zt))
            # print (f'\n\n\n')

            # print (torch.equal(xt[i, :context_lens[i]], x1[i, :context_lens[i]]))
            # print (torch.equal(xt[i, :context_lens[i]], zt[i, :context_lens[i]]))

        # mask where everything up to context_len is 0 for each element in batch
        max_len = xt.size(1)
        mask = torch.arange(max_len, device=self.device).unsqueeze(0) < context_lens.unsqueeze(1)
        mask = mask.float()  # Convert boolean mask to float (0.0 or 1.0)

        inverse_mask_expanded = (1 - mask).unsqueeze(-1)

        # rates = rates * inverse_mask_expanded
        # ins_probs = ins_probs * inverse_mask_expanded
        # sub_probs = sub_probs * inverse_mask_expanded


        lam_ins = rates[:, :, 0]
        lam_sub = rates[:, :, 1]
        lam_del = rates[:, :, 2]

        ux_ins = lam_ins.unsqueeze(-1) * ins_probs
        ux_sub = lam_sub.unsqueeze(-1) * sub_probs
        ux_del = lam_del.unsqueeze(-1)

        ux_cat = torch.cat([ux_ins, ux_sub, ux_del], dim=-1)

        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap, z_pad)

        uz_mask = make_ut_mask_from_z(zt, z1, self.full_vocab_size, self.pad_token, self.gap_token)

        u_tot = rates.sum(dim=(1, 2))
        # u_tot = rates.mean(dim=(1, 2))

        sched_coeff = (self.kappa.derivative(t) / (1 - self.kappa(t))).to(self.device)

        log_uz_cat = torch.clamp(uz_cat.log(), min=-20)

        term2  = (log_uz_cat * uz_mask * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))

        loss_vec = u_tot - term2

        # normalise by aligned sequence length
        N = torch.sum(~z_pad, dim=1).float()

        N = torch.clamp(N, min=1.0)

        loss_vec = loss_vec / N

        return loss_vec.mean(), {
            'utot': u_tot.mean(),
            'u_tot / N': (u_tot / N).mean(),
            '-term2': -term2.mean(),
            '-term2 / N': -(term2 / N).mean(),
            'u_ins': lam_ins.sum(1).mean(),
            'u_sub': lam_sub.sum(1).mean(),
            'u_del': lam_del.sum(1).mean(),
            'attn_mask_ratio': attn_mask_ratio,
            'N': N
        }

    def training_step(self, batch, batch_idx):
        try:
            loss, metrics = self._loss(batch)
            self.log('train/loss', loss, prog_bar=True)
            for k, v in metrics.items():
                self.log(f'train/{k}', v, prog_bar=False)
            return loss
        except Exception as e:
            print (f'Exception: {e}')
            torch.cuda.empty_cache()
            return None

    def backward(self, loss, *args, **kwargs):
        """
        This hook is called by Lightning after training_step.
        """
        # Ensure the flag is reset at the start of every backward call
        self._oom_in_backward = False

        try:
            # 2. Backward pass
            loss.backward()

        except torch.cuda.OutOfMemoryError:
            print(f"CUDA OOM in BACKWARD pass. Clearing gradients and skipping optimizer step.")
            # Set the flag to true
            self._oom_in_backward = True
            torch.cuda.empty_cache()

            # We must clear gradients here, otherwise they might be stale
            # for the next successful batch
            self.zero_grad(set_to_none=True)

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(
                    f"CUDA OOM (RuntimeError) in BACKWARD pass. Clearing gradients and skipping optimizer step.")
                self._oom_in_backward = True
                torch.cuda.empty_cache()
                self.zero_grad(set_to_none=True)
            else:
                raise e

    def optimizer_step(self, epoch, batch_idx, optimizer, *args, **kwargs):
        """
        This hook is called by Lightning after backward().
        """
        # 3. Optimizer step

        # Check the flag from the backward pass
        if self._oom_in_backward:
            print(f"Skipping optimizer step for batch {batch_idx} due to OOM in backward.")
            # Reset flag and skip step
            self._oom_in_backward = False
            return

            # If no OOM, proceed with the optimizer step
        super().optimizer_step(epoch, batch_idx, optimizer, *args, **kwargs)


    def validation_step(self, batch, batch_idx):
        loss, metrics = self._loss(batch)
        self.log('val/loss', loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(f'val/{k}', v, prog_bar=False)

        # # Select a few samples from the batch for generation
        # num_samples_to_log = min(4, batch['x0'].size(0))
        # for i in range(num_samples_to_log):
        #     x0_sample = batch['x0'][i].unsqueeze(0)
        #     context_len_sample = batch['context_lens'][i].unsqueeze(0)
        #
        #     # Generate a trajectory
        #     trajectory = self.sample(x0_sample, context_len_sample, n_steps=100)
        #
        #     # Log the initial and final states of the trajectory
        #     initial_seq = self.tokenizer.decode(trajectory[0].squeeze().tolist(), skip_special_tokens=False)
        #     final_seq = self.tokenizer.decode(trajectory[-1].squeeze().tolist(), skip_special_tokens=False)
        #
        #     print(f"Initial Sequence (Sample {i}): {initial_seq}")
        #     print(f"Final Sequence (Sample {i}): {final_seq}")
        #
        #     # self.logger.experiment.log({
        #     #     f"val/sample_{i}/initial_sequence": initial_seq,
        #     #     f"val/sample_{i}/final_sequence": final_seq,
        #     #     f"val/sample_{i}/trajectory_length": len(trajectory)
        #     # })
        #
        #     # # Optionally, log the full trajectory as a list of strings
        #     # full_trajectory_decoded = [self.tokenizer.decode(seq.squeeze().tolist(), skip_special_tokens=False) for seq in trajectory]
        #     # self.logger.experiment.log({
        #     #     f"val/sample_{i}/full_trajectory": wandb.Table(data=[[s] for s in full_trajectory_decoded], columns=["sequence"])
        #     # })

    # update sample to account for context (take in context and context length for a batch):
    @torch.no_grad()
    def sample(self, x0, context_lens, n_steps=100, t_min=0.0):
        self.model.eval()
        device = x0.device
        t = torch.full((x0.size(0), 1), t_min, device=device)
        dt = 1 / n_steps
        pad_mask = (x0 == self.pad_token)
        xt = x0.clone()
        traj = [xt.clone()]

        for _ in range(n_steps):
            # Create a mask for the context tokens
            max_len = xt.size(1)
            context_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < context_lens.unsqueeze(1)

            rates, ins_probs, sub_probs = self(xt, t, pad_mask)
            lam_i, lam_s, lam_d = rates[..., 0], rates[..., 1], rates[..., 2]

            # Apply context mask to rates
            lam_i = lam_i * (~context_mask)
            lam_s = lam_s * (~context_mask)
            lam_d = lam_d * (~context_mask)

            ins_mask = torch.rand_like(lam_i) < 1 - torch.exp(-dt * lam_i)
            ds_mask = torch.rand_like(lam_s) < 1 - torch.exp(-dt * (lam_s + lam_d))

            prob_del = torch.where(ds_mask, lam_d / (lam_s + lam_d + 1e-8), torch.zeros_like(lam_d))

            del_mask = torch.bernoulli(prob_del).bool()
            sub_mask = ds_mask & ~del_mask
            non_pad = ~pad_mask

            ins_tokens = torch.full_like(xt, self.pad_token)
            sub_tokens = torch.full_like(xt, self.pad_token)

            if non_pad.any():
                # Ensure we only sample for non-context tokens
                ins_tokens[non_pad & ~context_mask[non_pad]] = torch.multinomial(
                    ins_probs[non_pad & ~context_mask[non_pad]], 1).squeeze(-1)
                sub_tokens[non_pad & ~context_mask[non_pad]] = torch.multinomial(
                    sub_probs[non_pad & ~context_mask[non_pad]], 1).squeeze(-1)

            # Apply operations only to non-context tokens
            xt[sub_mask & ~context_mask] = sub_tokens[sub_mask & ~context_mask]
            xt[del_mask & ~context_mask] = self.pad_token

            for b in range(xt.size(0)):
                # Only consider insertions for non-context tokens
                positions = torch.nonzero(ins_mask[b] & ~context_mask[b], as_tuple=True)[0]
                for p in positions.tolist():
                    # Find a pad position *outside* the context to insert into
                    # This is a simplified approach; a more robust solution might involve shifting
                    # or finding the nearest available non-context pad token.
                    # For now, we'll try to insert into any pad token, but the ins_mask ensures
                    # the *decision* to insert came from a non-context token.
                    pad_pos = (xt[b] == self.pad_token).nonzero(as_tuple=True)[0]
                    if pad_pos.numel():
                        # Find the first pad position that is also outside the context
                        valid_insert_pos = [pos for pos in pad_pos if not context_mask[b, pos]]
                        if valid_insert_pos:
                            tgt = valid_insert_pos[0]
                            xt[b, tgt] = ins_tokens[b, p]

            pad_mask = (xt == self.pad_token)
            t = t + dt
            traj.append(xt.clone())

        return traj

        # todo validation testing
        # todo ensure all samples keep context assertions, fix + test context masking
        # todo log edit distance

        # todo add wandb / bait integration
        # todo for future data gen runs, filter to be less than certain length to save time
        # todo maybe try just training with error correction only?
