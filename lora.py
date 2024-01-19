from peft import TaskType
from transformers import BitsAndBytesConfig
import torch


def get_lora_configuration(model_name: str) -> dict:
    if "distil" in model_name:
        target_modules = {
            "q_lin",
            "k_lin",
            "v_lin",
            "out_lin",
        }
    else:
        target_modules = {
            "query",
            "key",
            "value",
            "dense",
        }
    lora_dict = dict(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        target_modules=target_modules,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    return lora_dict


def get_quantization_configuration() -> BitsAndBytesConfig:
    four_bit_quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return four_bit_quantization
