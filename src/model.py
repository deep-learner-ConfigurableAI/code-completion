"""
Core model implementation for code completion.
"""

import functools
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

from settings import (
    MODEL, LORA_ALPHA, LORA_DROPOUT, LORA_R, LORA_TARGET_MODULES,
    USE_NESTED_QUANT, BNB_4BIT_COMPUTE_DTYPE, DEVICE, SEED
)


@functools.lru_cache(maxsize=None)
def get_fim_token_ids(tokenizer):
    """Get the FIM token IDs from the tokenizer."""
    try:
        FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD = tokenizer.special_tokens_map["additional_special_tokens"][1:5]
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = (
            tokenizer.vocab[tok] for tok in [FIM_SUFFIX, FIM_PREFIX, FIM_MIDDLE, FIM_PAD]
        )
    except KeyError:
        suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id = None, None, None, None
    return suffix_tok_id, prefix_tok_id, middle_tok_id, pad_tok_id


def permute(
    sample,
    np_rng,
    suffix_tok_id,
    prefix_tok_id,
    middle_tok_id,
    pad_tok_id,
    fim_rate=0.5,
    fim_spm_rate=0.5,
    truncate_or_pad=False,
):
    """
    Take in a sample (list of tokens) and perform a FIM transformation on it with a probability of fim_rate, using two FIM modes:
    PSM and SPM (with a probability of fim_spm_rate).
    """
    # The if condition will trigger with the probability of fim_rate
    # This means FIM transformations will apply to samples with a probability of fim_rate
    if np_rng.binomial(1, fim_rate):

        # Split the sample into prefix, middle, and suffix, based on randomly generated indices stored in the boundaries list.
        boundaries = list(np_rng.randint(low=0, high=len(sample) + 1, size=2))
        boundaries.sort()

        prefix = np.array(sample[: boundaries[0]], dtype=np.int64)
        middle = np.array(sample[boundaries[0] : boundaries[1]], dtype=np.int64)
        suffix = np.array(sample[boundaries[1] :], dtype=np.int64)

        if truncate_or_pad:
            # calculate the new total length of the sample, taking into account tokens indicating prefix, middle, and suffix
            new_length = suffix.shape[0] + prefix.shape[0] + middle.shape[0] + 3
            diff = new_length - len(sample)

            # trancate or pad if there's a difference in length between the new length and the original
            if diff > 0:
                if suffix.shape[0] <= diff:
                    return sample, np_rng
                suffix = suffix[: suffix.shape[0] - diff]
            elif diff < 0:
                suffix = np.concatenate([suffix, np.full((-1 * diff), pad_tok_id)])

        # With the probability of fim_spm_rateapply SPM variant of FIM transformations
        # SPM: suffix, prefix, middle
        if np_rng.binomial(1, fim_spm_rate):
            new_sample = np.concatenate(
                [
                    [prefix_tok_id, suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    prefix,
                    middle,
                ]
            )
        # Otherwise, apply the PSM variant of FIM transformations
        # PSM: prefix, suffix, middle
        else:
            new_sample = np.concatenate(
                [
                    [prefix_tok_id],
                    prefix,
                    [suffix_tok_id],
                    suffix,
                    [middle_tok_id],
                    middle,
                ]
            )
    else:
        # don't apply FIM transformations
        new_sample = sample

    return list(new_sample), np_rng


def example_generator(dataset, content_field="content"):
    """Generator function to yield examples from a dataset."""
    for example in dataset:
        yield example[content_field]


class ConstantLengthDatasetAsDataset(Dataset):
    """Dataset that returns fixed-length chunks from the input data."""
    def __init__(
        self,
        tokenizer,
        dataset,
        seq_length=1024,
        chars_per_token=2.6,
        content_field="content",
        fim_rate=0.5,
        fim_spm_rate=0.5,
        seed=0,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.content_field = content_field
        self.fim_rate = fim_rate
        self.fim_spm_rate = fim_spm_rate
        self.seed = seed
        self.total_tokens = 0

        self.examples = []
        np_rng = np.random.RandomState(seed=self.seed)

        (
            self.suffix_tok_id,
            self.prefix_tok_id,
            self.middle_tok_id,
            self.pad_tok_id,
        ) = get_fim_token_ids(self.tokenizer)
        if not self.suffix_tok_id and self.fim_rate > 0:
            print("FIM is not supported by tokenizer, disabling FIM")
            self.fim_rate = 0

        gen = example_generator(self.dataset, content_field=self.content_field)

        all_token_ids = []
        for text in gen:
            tokenized_input = self.tokenizer(text, truncation=False)["input_ids"]
            if self.fim_rate > 0:
                tokenized_input, np_rng = permute(
                    tokenized_input,
                    np_rng,
                    self.suffix_tok_id,
                    self.prefix_tok_id,
                    self.middle_tok_id,
                    self.pad_tok_id,
                    fim_rate=self.fim_rate,
                    fim_spm_rate=self.fim_spm_rate,
                    truncate_or_pad=False,
                )
            all_token_ids.extend(tokenized_input + [self.concat_token_id])

        self.total_tokens += len(all_token_ids)

        for i in range(0, len(all_token_ids), self.seq_length):
            input_ids = all_token_ids[i : i + self.seq_length]
            if len(input_ids) == self.seq_length:
                self.examples.append({
                    "input_ids": torch.LongTensor(input_ids),
                    "labels": torch.LongTensor(input_ids),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_model_tokenizer(model_id=MODEL):
    """Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    load_in_8bit = False
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=load_in_8bit,
        device_map="auto",
        use_cache=False,  # We will be using gradient checkpointing
        trust_remote_code=True,
        use_flash_attention_2=False,
    )
    
    return model, tokenizer


def create_peft_model(model):
    """Create a PEFT model with LoRA configuration."""
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES.split(","),
    )

    model = get_peft_model(model, peft_config)
    return model


def chars_token_ratio(dataset, tokenizer, data_column, nb_examples=400):
    """Estimate the average number of characters per token in the dataset."""
    from tqdm import tqdm
    
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        total_characters += len(example[data_column])
        total_tokens += len(tokenizer(example[data_column]).tokens())

    return total_characters / total_tokens


def get_code_completion(model, tokenizer, prefix, suffix, max_new_tokens=128):
    """Generate code completion given a prefix and suffix."""
    text = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
    model.eval()
    outputs = model.generate(
        input_ids=tokenizer(text, return_tensors="pt").input_ids.to(DEVICE),
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.0,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
