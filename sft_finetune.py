import logging
import sys
from typing import Any, Dict, List
import os
import datasets
import determined as det
import transformers
from datasets import DatasetDict, concatenate_datasets, load_dataset
from determined.transformers import DetCallback
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from huggingface_hub import snapshot_download

from utils import get_model, get_tokenizer

logger = logging.getLogger(__name__)

os.environ["HF_TOKEN"] = ""
hf_cache_dir = "/pfss/mlde/workspaces/mlde_wsp_P_AvemioAG/soumya_test/cache"

def load_sft_dataset(hparams: Dict[str, Any]) -> DatasetDict:
    dataset_name = hparams["dataset"]
    dataset_subsets = hparams["dataset_subsets"]
    train_datasets = []
    test_datasets = []
    for subset_info in dataset_subsets:
        if "ratio" in subset_info:
            subset_str = f"{int(subset_info['ratio']*100)}%"
        elif "number_of_samples" in subset_info:
            subset_str = str(subset_info["number_of_samples"])
        else:
            raise RuntimeError(f"Unknown subset definition {subset_info}")
        train_dataset = load_dataset(
            dataset_name, subset_info["subset"], split=f"train[:{subset_str}]", cache_dir=hf_cache_dir
        )
        train_datasets.append(train_dataset)

        test_dataset = load_dataset(
            dataset_name, subset_info["subset"], split=f"test", cache_dir=hf_cache_dir
        )
        test_datasets.append(test_dataset)

    final_train_dataset = concatenate_datasets(train_datasets)
    final_test_dataset = concatenate_datasets(test_datasets)
    dataset = DatasetDict({'train': final_train_dataset, 'test': final_test_dataset})
    
    return dataset


def setup_special_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    special_tokens: List[str],
):
    # https://github.com/huggingface/trl/blob/66078c7c0142c7aada994856151e7e22745d4ecf/trl/models/utils.py#L43
    # We won't be changing bos, eos and pad token, though.
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    return model, tokenizer



def formatting_prompts_func(example, tokenizer):
    output_texts = []
    total_systems = len(example["system"])
    
    for index in range(len(example["system"])):
        conversation=example["conversations"][index]
        flag_function=False
        system_content=example ["system"][index]

         # Initialize contents
        prompt = [{"role": "system", "content": system_content}]
        
        for item in conversation:
            role = item["from"] 
            content = item["value"]  
            
            if role == "human":
                prompt.append({"role": "user", "content": content})
            elif role == "gpt":
                prompt.append({"role": "assistant", "content": content})
            elif role == "function_call":
                prompt.append({"role": "assistant", "content": "<tool_call>\n" + content + "\n</tool_call>"})
            elif role == "observation":
                prompt.append({"role": "user", "content": "<tool_response>\n"+ content + "\n</tool_response>"})
    
        try:
            text = tokenizer.apply_chat_template(prompt, tokenize=False)
            output_texts.append(text)
        except Exception as e:
            logging.error(f"Error formatting prompt: {e}")
            output_texts.append("")
              

    return output_texts 

def main(training_args, det_callback, hparams):
    set_seed(training_args.seed)
    dataset = load_sft_dataset(hparams)

    model_name = hparams["model"]
    if core_context.distributed.get_local_rank() == 0:
        snapshot_download(model_name, cache_dir=hf_cache_dir)
    _ = core_context.distributed.broadcast_local()    
    model = get_model(model_name, hf_cache_dir)
    tokenizer = get_tokenizer(
        model_name,
        padding_side="right",
        truncation_side="right",
        model_max_length=hparams["max_seq_length"],
        add_eos_token=True,
    )

    if hparams["chat_tokens"]["add_chat_tokens"]:
        model, tokenizer = setup_special_tokens(
            model, tokenizer, hparams["chat_tokens"]["special_tokens"]
        )
    #    dataset = dataset.map(lambda example: formatting_prompts_func(example, tokenizer), batched=True)

    logging.info(f"dataset_sample={dataset['train'][0]}")
    
    if hparams["data_collator"]["on_completions_only"]:
        assistant_prompt = hparams["data_collator"]["response_template"]
        response_template_ids = tokenizer.encode(
            assistant_prompt, add_special_tokens=False
        )
        collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=tokenizer
        )
        logging.info("Using DataCollatorForCompletionOnlyLM.")
    else:
        collator = None
        logging.info("Using default data collator")

    trainer = SFTTrainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=lambda example: formatting_prompts_func(example, tokenizer),
        max_seq_length=hparams["max_seq_length"],
    )


# Update the map function to use the correct field
#dataset = dataset.map(formatting_prompts_func, batched=True)



    trainer.add_callback(det_callback)

    trainer.train()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format=det.LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    log_level = logging.INFO
    transformers.utils.logging.set_verbosity_info()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    info = det.get_cluster_info()
    hparams = info.trial.hparams

    training_args = TrainingArguments(**hparams["training_args"])
    distributed = det.core.DistributedContext.from_deepspeed()
    with det.core.init(distributed=distributed) as core_context:
        det_callback = DetCallback(
            core_context,
            training_args,
        )
        main(training_args, det_callback, hparams)
