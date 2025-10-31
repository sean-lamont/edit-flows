import torch
import matplotlib.pyplot as plt
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from utils import SinusoidalTimeEmbedding

class AdaptedEditFlowsTransformer(nn.Module):
    def __init__(self, pretrained_model_name: str, hidden_dim=512, debug_attn=False):
        super().__init__()

        self.debug_attn = debug_attn

        # bnb_conf = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, dtype=torch.bfloat16,
                                                          trust_remote_code=True,
                                                          # _attn_implementation='flex_attention',
                                                          # _attn_implementation='flash_attention_2',
                                                          # _attn_implementation='eager'
                                                          # quantization_config=bnb_conf,
                                                          output_attentions=True,
                                                          ).train()

        # self.model = prepare_model_for_kbit_training(self.model)

        # add LoRa and Quantization

        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
            lora_dropout=0.05,
            # bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # self.model.gradient_checkpointing_enable()
        # self.model.config.use_cache = False

        # self.model.compile()

        self.vocab_size = self.model.config.vocab_size
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)

        self.rate_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, 3)) # 3 for ins,sub,del, extra 2 for weighting lm_head

        self.ins_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, self.model.config.hidden_size), nn.SiLU(),
                                      nn.Linear(self.model.config.hidden_size, self.vocab_size))

        self.sub_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, self.model.config.hidden_size), nn.SiLU(),
                                      nn.Linear(self.model.config.hidden_size, self.vocab_size))


    def forward(self, tokens: torch.Tensor, t: torch.Tensor, pad_mask: torch.Tensor, context_tokens, pad_token,
                attn_mask_ratio: float = 0.0):

        context_lens = [c.shape[0] for c in context_tokens]
        pad_len = tokens.shape[1] + max(context_lens)

        active_target_lens = (~pad_mask).sum(dim=1)


        combined_tokens = torch.stack([
            F.pad(
                torch.cat([context_tokens[i], tokens[i]], dim=0),
                (0, pad_len - (tokens.shape[1] + context_lens[i])), value=pad_token)
            for i in range(tokens.shape[0])], dim=0).long()

        B, L = combined_tokens.shape

        # Create the base causal mask (True = keep)
        # This is [L, L], which will be broadcast to [B, 1, L, L]
        causal_bool = torch.tril(torch.ones((L, L), device=tokens.device, dtype=torch.bool))

        # Create the annealed random mask (True = keep)
        # This is [B, 1, L, L] for a different mask per batch item
        # random_bool = torch.rand(B, 1, L, L, device=tokens.device) < attn_mask_ratio
        random_bool = torch.rand(B, 1, L, L, device=tokens.device) < 0.1

        # pad_locations is [B, L], True where there is a pad
        pad_locations = (combined_tokens == pad_token)

        # We want to mask out any position where
        # the query (row) or the key (column) is a pad token.
        # [B, L] -> [B, 1, L, 1] (for queries)
        # [B, L] -> [B, 1, 1, L] (for keys)
        # OR broadcasts to [B, 1, L, L]
        # will be True where either q or k is pad
        padding_mask_bool = pad_locations.view(B, 1, L, 1) | pad_locations.view(B, 1, 1, L)
        padding_bool = ~padding_mask_bool

        # We want to keep positions that are:
        # (Causal OR Random) AND (Not Padding)
        final_mask_bool = (causal_bool | random_bool) & padding_bool

        # attention_mask = ~final_mask_bool

        # 5. Convert boolean mask to a float mask for the model
        # The model's forward pass expects 0.0 for "keep" and -inf for "mask out"
        attention_mask = torch.zeros(B, 1, L, L, device=tokens.device, dtype=torch.bfloat16)

        attention_mask.masked_fill_(~final_mask_bool, -torch.inf)

        outputs = self.model.forward(input_ids=combined_tokens,
                                     attention_mask=attention_mask,  # <-- Pass the dense mask here
                                     output_hidden_states=True,
                                     output_attentions=True)

        if self.debug_attn:
            # We'll just plot for the first item in the batch (B=0)
            context_len_b0 = context_lens[0]
            active_target_len_b0 = active_target_lens[0].item()

            last_layer_attn = outputs.attentions[-1]
            L_combined = last_layer_attn.shape[-1]

            # Get attention for Batch 0, Head 0
            attn_map = last_layer_attn[0, 0].detach().cpu().float().numpy()

            plt.figure(figsize=(12, 10))
            plt.imshow(attn_map, vmin=0)
            plt.title(f"[Debug Plot] Attn Map (B=0, H=0) | Ratio: {attn_mask_ratio}")
            plt.xlabel("Key Position (Attending To)")
            plt.ylabel("Query Position (Attending From)")
            plt.colorbar(label="Attention Weight")

            # Line 1: End of Context
            context_end_line = context_len_b0 - 0.5
            plt.axvline(x=context_end_line, color='cyan', linestyle='--', label='Context / Target Boundary')
            plt.axhline(y=context_end_line, color='cyan', linestyle='--')

            # Line 2: End of Active Target (Start of Padding)
            active_target_end_line = (context_len_b0 + active_target_len_b0) - 0.5
            plt.axvline(x=active_target_end_line, color='r', linestyle='--', label='Target / Padding Boundary')
            plt.axhline(y=active_target_end_line, color='r', linestyle='--')

            plt.xlim(-0.5, L_combined - 0.5)
            plt.ylim(L_combined - 0.5, -0.5)
            plt.legend()

            print("\n--- [Debug Plot] ---")
            print(f"Plotting attention for B=0, H=0 (Shape: {L_combined}x{L_combined})")
            print(f"  Context len: {context_len_b0}")
            print(f"  Active Target len: {active_target_len_b0}")
            print("--- Pausing script. Close plot window to continue. ---")

            plt.savefig("attention_plot_causal.png", dpi=300, bbox_inches='tight')

        # (b, seq_len, dim)
        hidden_states = outputs.hidden_states[-1]

        # only take hidden states from context_lens onwards
        hidden_states = torch.stack([
            hidden_states[i, context_lens[i]:context_lens[i] + tokens.shape[1]]  # same length (original padded non-context tokens, taken from respective context )
            for i in range(hidden_states.shape[0])], dim=0 )

        time_ = self.time_emb(t).unsqueeze(1).expand(-1, hidden_states.shape[1], -1)

        x = torch.cat([hidden_states, time_], dim=-1)

        # (b, seq, 5)
        rates = F.softplus(self.rate_head(x))

        # (b, seq, vocab)
        # lm_output = self.model.lm_head(hidden_states)

        # add zero vector for first entry
        # lm_output = torch.cat([torch.zeros_like(lm_output[:, 0], device=x.device, dtype=torch.bfloat16).unsqueeze(1), lm_output], dim=1)

        # add lm output for insert (usual objective from pretrained model), previous token as substitute, scaled by learned weighting params
        # ins = F.softmax(self.ins_head(x) + rates[:, :, -1].unsqueeze(-1) * lm_output[:, 1:], dim=-1)
        # sub = F.softmax(self.sub_head(x) + rates[:, :, -2].unsqueeze(-1) * lm_output[:, :-1], dim=-1)

        ins = F.softmax(self.ins_head(x), dim=-1)
        sub = F.softmax(self.sub_head(x), dim=-1)

        mask = (~pad_mask).unsqueeze(-1).float()

        # return (rates[:, :, :3] * mask, ins * mask, sub * mask)
        return (rates * mask, ins * mask, sub * mask)
