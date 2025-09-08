"""
Training and execution logic for the code completion model.
"""

import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import PeftModel

from settings import (
    MODEL, DATASET, DATA_COLUMN, SEQ_LENGTH, 
    MAX_STEPS, BATCH_SIZE, GR_ACC_STEPS, LR, LR_SCHEDULER_TYPE,
    WEIGHT_DECAY, NUM_WARMUP_STEPS, EVAL_FREQ, SAVE_FREQ, LOG_FREQ,
    OUTPUT_DIR, BF16, FP16, FIM_RATE, FIM_SPM_RATE, SEED, HF_TOKEN, DEVICE
)
from model import (
    load_model_tokenizer, create_peft_model, ConstantLengthDatasetAsDataset,
    chars_token_ratio, get_code_completion
)


def setup_environment():
    """Set up the environment for training and inference."""
    set_seed(SEED)
    os.environ["HF_TOKEN"] = HF_TOKEN
    
    
def load_dataset_splits():
    """Load and prepare the dataset splits."""
    dataset = load_dataset(DATASET, data_dir="data", split="train", streaming=True)
    
    valid_data = dataset.take(2000)
    train_data = dataset.skip(2000)
    train_data = train_data.shuffle(buffer_size=5000, seed=SEED)
    
    return train_data, valid_data


def prepare_datasets(tokenizer, train_data, valid_data):
    """Create training and evaluation datasets."""
    # Calculate characters per token ratio
    chars_per_token = chars_token_ratio(train_data, tokenizer, DATA_COLUMN)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")
    
    # Create datasets
    train_dataset = ConstantLengthDatasetAsDataset(
        tokenizer,
        train_data,
        seq_length=SEQ_LENGTH,
        chars_per_token=chars_per_token,
        content_field=DATA_COLUMN,
        fim_rate=FIM_RATE,
        fim_spm_rate=FIM_SPM_RATE,
        seed=SEED,
    )
    
    eval_dataset = ConstantLengthDatasetAsDataset(
        tokenizer,
        valid_data,
        seq_length=SEQ_LENGTH,
        chars_per_token=chars_per_token,
        content_field=DATA_COLUMN,
        fim_rate=FIM_RATE,
        fim_spm_rate=FIM_SPM_RATE,
        seed=SEED,
    )
    
    return train_dataset, eval_dataset


def setup_training(model, train_dataset, eval_dataset):
    """Set up the training configuration and trainer."""
    training_args = TrainingArguments(
        output_dir=f"verma75preetam/{OUTPUT_DIR}",
        dataloader_drop_last=True,
        do_eval=True,
        save_strategy="steps",
        max_steps=MAX_STEPS,
        eval_steps=EVAL_FREQ,
        save_steps=SAVE_FREQ,
        logging_steps=LOG_FREQ,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_steps=NUM_WARMUP_STEPS,
        gradient_accumulation_steps=GR_ACC_STEPS,
        gradient_checkpointing=True,
        fp16=FP16,
        bf16=BF16,
        weight_decay=WEIGHT_DECAY,
        push_to_hub=True,
        include_tokens_per_second=True,
    )
    
    model.train()
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset
    )
    
    return trainer


def train_model():
    """Full training pipeline."""
    # Setup
    setup_environment()
    
    # Load data
    train_data, valid_data = load_dataset_splits()
    
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer()
    
    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(tokenizer, train_data, valid_data)
    
    # Apply LoRA
    model = create_peft_model(model)
    
    # Print model information
    model.print_trainable_parameters()
    
    # Count trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters count: {len(trainable_params)}")
    
    # Optionally print names and shapes
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")
    
    # Set up training
    trainer = setup_training(model, train_dataset, eval_dataset)
    
    # Train
    print("Training...")
    trainer.train()
    
    print("Training complete!")


def load_model_for_inference():
    """Load a trained model for inference."""
    # Step 1: Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(MODEL)
    
    # Step 2: Load LoRA adapter on top
    model = PeftModel.from_pretrained(base_model, f"verma75preetam/{OUTPUT_DIR}")
    
    # Move models to device
    model.to(DEVICE)
    base_model.to(DEVICE)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    return model, tokenizer


def code_completion_demo():
    """Demo for code completion."""
    model, tokenizer = load_model_for_inference()
    
    prefix = """def tokenize_function(examples):
"""
    suffix = """"""
    
    completion = get_code_completion(model, tokenizer, prefix, suffix)
    print(completion)


if __name__ == "__main__":
    # Uncomment the function you want to run
    # train_model()
    # code_completion_demo()
    pass
