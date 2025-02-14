import logging
import os
from typing import Optional

import determined as det
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.trainer_utils import get_last_checkpoint

from chat_format import *


def get_model(model_name, model_cache_dir, inference=False, device_map="auto") -> PreTrainedModel:
    
    if inference:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=model_cache_dir,
            local_files_only=True
        )
    return model


def get_tokenizer(
    model_name: str,
    padding_side: str = "right",
    truncation_side: str = "right",
    model_max_length: Optional[int] = None,
    add_eos_token: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        truncation_side=truncation_side,
        padding_side=padding_side,
    )
   # if tokenizer.pad_token_id is None:
   #     tokenizer.pad_token_id = tokenizer.eos_token_id

   # if tokenizer.model_max_length > 100_000:
   #     tokenizer.model_max_length = model_max_length if model_max_length else 2048
    tokenizer.model_max_length=model_max_length

   #if tokenizer.chat_template is None:
    tokenizer.chat_template = CHAT_ML_Phi

    # For Phi model you must use it
    special_tokens = ["<|user|>", "<|system|>","<|assistant|>","<|end|>"]
    # For all models except Phi models
    #special_tokens = ["<|im_start|>", "<|im_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    tokenizer.add_eos_token = add_eos_token
    return tokenizer


def download_ckpt(ckpt_uuid: str, core_context: det.core.Context) -> str:
    download_dir = os.path.join(os.environ.get("HF_CACHE", "."), ckpt_uuid)

    def selector(path: str) -> bool:
        if any(
            [
                path.endswith(ext)
                for ext in [
                    "config.json",
                    "generation-config.json",
                    ".safetensors",
                    "special_tokens_map.json",
                    "tokenizer_config.json",
                    "tokenizer.json",
                    "tokenizer.model",
                    "model.safetensors.index.json",
                ]
            ]
        ):
            return True

        return False

    core_context.checkpoint.download(ckpt_uuid, download_dir, selector=selector)
    model_dir = get_last_checkpoint(download_dir)
    return model_dir
