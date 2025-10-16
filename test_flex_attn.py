from outlines.models import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.attention.flex_attention import create_block_mask, and_masks, or_masks, create_mask
from transformers.utils import logging, TransformersKwargs

logger = logging.get_logger(__name__)


model_id = "Goedel-LM/Goedel-Prover-V2-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=True, _attn_implementation='flex_attention')

formal_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat


theorem square_equation_solution {x y : ℝ} (h : x^2 + y^2 = 2*x - 4*y - 5) : x + y = -1 := by
  sorry
""".strip()

prompt = """
Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

chat = [
    [{"role": "user", "content": prompt.format(formal_statement)}],
    [{'role': 'user', 'content': prompt.format(formal_statement) + 'asdfasdf'}]
]

inputs = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, padding='longest',
                                       return_dict=True, return_tensors='pt', max_length=128, truncation=True)

def causal_mask(b,h,q_idx,kv_idx):
    return q_idx >= kv_idx

def create_padding_mask(pads):
    def padding(b, h, q_idx, kv_idx):
        return ~pads[b, q_idx] & ~pads[b, kv_idx]
    return padding


attn_mask_ratio = 0.5
seq_len = inputs['input_ids'].shape[1]

random_mask = (torch.rand(seq_len, seq_len) < attn_mask_ratio).to('cuda')
pads = inputs['attention_mask'].to('cuda')
padding_mask = create_padding_mask(pads)

def create_random_mask(b,h,q_idx,kv_idx):
    return random_mask[q_idx][kv_idx]


full_mask = or_masks(*[causal_mask, create_random_mask])


block_mask = create_block_mask(causal_mask, None, None, len(inputs['input_ids'][0]), len(inputs['input_ids'][0]), device='cuda', _compile=True)

print (block_mask)
model.eval()
model.cuda()
model.compile()

# output_attnetions for flex_attention outputs only logsumexp of (bsize, nheads, q) rather than (bsize, nheads, q, k)..
# https://github.com/huggingface/transformers/issues/36096

# error with triton, requires manually setting block sizes:
# https://github.com/pytorch/pytorch/issues/133254
with torch.no_grad():
    out = model.forward(inputs['input_ids'].to('cuda'), attention_mask=block_mask, output_hidden_states=True, output_attentions=True, kernel_options={
    "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_M1": 32,
        "BLOCK_N1": 64,
        "BLOCK_M2": 64,
        "BLOCK_N2": 32,
                        })

#                     kernel_options={
# "BLOCK_M": 64,
#     "BLOCK_N": 64,
#     "BLOCK_M1": 32,
#     "BLOCK_N1": 64,
#     "BLOCK_M2": 64,
#     "BLOCK_N2": 32,
#                     })

print(out.attentions[0][0][2][0])
