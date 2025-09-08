"""
Configuration settings for the code completion model.
"""

# Model configuration
MODEL = "bigcode/starcoderbase-1b"  # Model checkpoint on the Hugging Face Hub
DATASET = "smangrul/hf-stack-v1"    # Dataset on the Hugging Face Hub
DATA_COLUMN = "content"             # Column name containing the code content
SEQ_LENGTH = 1024                   # Sequence length

# Training arguments
MAX_STEPS = 2000                    # max_steps
BATCH_SIZE = 4                      # batch_size
GR_ACC_STEPS = 2                    # gradient_accumulation_steps
LR = 5e-4                           # learning_rate
LR_SCHEDULER_TYPE = "cosine"        # lr_scheduler_type
WEIGHT_DECAY = 0.01                 # weight_decay
NUM_WARMUP_STEPS = 30               # num_warmup_steps
EVAL_FREQ = 100                     # eval_freq
SAVE_FREQ = 100                     # save_freq
LOG_FREQ = 25                       # log_freq
OUTPUT_DIR = "peft-starcoder-lora-apple"  # output_dir
BF16 = True                         # bf16
FP16 = False                        # no_fp16

# FIM transformations arguments
FIM_RATE = 0.5                      # fim_rate
FIM_SPM_RATE = 0.5                  # fim_spm_rate

# LORA
LORA_R = 8                          # lora_r
LORA_ALPHA = 32                     # lora_alpha
LORA_DROPOUT = 0.0                  # lora_dropout
LORA_TARGET_MODULES = "c_proj,c_attn,q_attn,c_fc,c_proj"  # lora_target_modules

# bitsandbytes config
USE_NESTED_QUANT = True             # use_nested_quant
BNB_4BIT_COMPUTE_DTYPE = "bfloat16" # bnb_4bit_compute_dtype

# Random seed
SEED = 0

# Device
DEVICE = "mps"

# HuggingFace token
HF_TOKEN = ""  # Add your Hugging Face token here if you want to push the model to the Hub
