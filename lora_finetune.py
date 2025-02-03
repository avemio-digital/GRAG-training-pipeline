import logging
import sys
from typing import Any, Dict, List

import pandas as pd
import datasets
import determined as det
import transformers
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from determined.transformers import DetCallback
from transformers import PreTrainedTokenizer, TrainingArguments, set_seed
from trl import ORPOConfig, ORPOTrainer
import os

from lora_utils import *
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

TRIPLET_DATASET = "conversations,chosen,rejected"
CONVERSATION_DATASET = "chosen,rejected"

os.environ["HF_TOKEN"] = ""
hf_cache_dir = "/pfss/mlde/workspaces/mlde_wsp_P_AvemioAG/soumya_test/cache"


def is_feature_chat_conversation_format(dataset: Dataset, feature: str) -> bool:
    example = dataset[0][feature]
    if isinstance(example, list) and all(isinstance(x, dict) for x in example):
        for sample in example:
            if "content" not in sample or "role" not in sample:
                raise RuntimeError(
                    f"Column {feature} has data in unsupported format : {sample}"
                )
        return True
    else:
        raise RuntimeError(
            f"Column {feature} has data in unsupported format : {example}"
        )


def get_dataset_format(dataset: Dataset) -> str:
    if "chosen" not in dataset.features or "rejected" not in dataset.features:
        raise RuntimeError(
            f"DPO-compatible dataset requires 'chosen' and 'rejected' features."
        )

    if all(feature in dataset.features for feature in ["conversations", "chosen", "rejected"]):
        return TRIPLET_DATASET

    if is_feature_chat_conversation_format(
        dataset, "chosen"
    ) and is_feature_chat_conversation_format(dataset, "rejected"):
        return CONVERSATION_DATASET


def process_triplet_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizer
) -> Dataset:
    def apply_chat_template(example):
        
        for conv in example["conversations"]:
            if conv["from"] == "system":
                prompt_system = conv["value"]  
            else:
                prompt_user =conv["value"] 

        if prompt_system:
            example["prompt"] = tokenizer.apply_chat_template(
            [{"role": "system", "content": prompt_system}],
            tokenize=False,)+tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_user}],
            tokenize=False,)
        else:
            example["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_user}],
            tokenize=False,)
        
        example["chosen"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": example["chosen"]['value']}], tokenize=False
        )
        example["rejected"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": example["rejected"]['value']}], tokenize=False
        )
        return example

    columns = set(dataset.features) - {"prompt", "rejected", "chosen"}
    dataset = dataset.map(apply_chat_template, remove_columns=list(columns))
    return dataset


def load_dpo_datasets(
    datasets: List[str], tokenizer: PreTrainedTokenizer
) -> DatasetDict:
    dataset_list_validated = []
    dataset_subsets = hparams["dataset_subsets"]

    for dataset_name in dataset_subsets:
        subset_str = str(dataset_name["number_of_samples"])
        dataset = load_dataset(datasets,dataset_name["subset"],split=f"train[:{subset_str}]",cache_dir=hf_cache_dir)
        if isinstance(dataset, DatasetDict):
            dataset_list = [dataset[k] for k in dataset]
        else:
            dataset_list = [dataset]

        for ds in dataset_list:
            dataset_format = get_dataset_format(ds)
            if dataset_format == CONVERSATION_DATASET:
                ds = process_conversation_dataset(ds, tokenizer)
            elif dataset_format == TRIPLET_DATASET:
                ds = process_triplet_dataset(ds, tokenizer)
            #print("process dataset")
            #print(ds[:3])
            dataset_list_validated.append(ds)

    dataset = concatenate_datasets(dataset_list_validated)
    print("concatenate dataset")
    print(dataset[:3])
    dataset = dataset.train_test_split(test_size=0.01, shuffle=False)
    return dataset



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

def setup_special_tokens(
    model,
    tokenizer
):

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    return model, tokenizer



def main(training_args, det_callback, hparams):
    set_seed(training_args.seed)

    model_name = hparams["model"]
    if core_context.distributed.get_local_rank() == 0:
        snapshot_download(model_name, cache_dir=hf_cache_dir)
    _ = core_context.distributed.broadcast_local()  
    model = get_model_lora(model_name,model_cache_dir=hf_cache_dir)
    tokenizer = get_tokenizer(
        model_name,
        truncation_side="right",
        padding_side="right",
        model_max_length=hparams['training_args']["max_length"],
        add_eos_token=False,
    )

    model, tokenizer = setup_special_tokens(
            model, tokenizer
        )

    
    dataset =load_dpo_datasets(hparams["dataset"], tokenizer)
    #df = pd.DataFrame(dataset['train'][:])
    #df_sample = df.sample(n=500, random_state=42)
    
    #df.to_csv('/pfss/mlde/workspaces/mlde_wsp_P_AvemioAG/soumya_test/output/ORPO_train_dataset_v1.csv',index=False)


    #model, tokenizer = setup_special_tokens(
           # model, tokenizer
        #)


    trainer = ORPOTrainer(
        model=model,
        args= training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer
    )
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

    training_args = ORPOConfig(**hparams["training_args"])
    distributed = det.core.DistributedContext.from_deepspeed()
    with det.core.init(distributed=distributed) as core_context:
        det_callback = DetCallback(
            core_context,
            training_args,
        )
        main(training_args, det_callback, hparams)
