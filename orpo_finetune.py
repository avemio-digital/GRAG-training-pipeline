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

from utils import download_ckpt, get_model, get_tokenizer
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

TRIPLET_DATASET = "conversation,chosen,rejected"
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

    if all(feature in dataset.features for feature in ["conversation", "chosen", "rejected"]):
        return TRIPLET_DATASET

    if is_feature_chat_conversation_format(
        dataset, "chosen"
    ) and is_feature_chat_conversation_format(dataset, "rejected"):
        return CONVERSATION_DATASET


def process_conversation_dataset(dataset: Dataset, tokenizer) -> Dataset:
    processed_data = {"prompt": [], "chosen": [], "rejected": []}

    for example in dataset:
        assert ". ".join([x["content"] for x in example["chosen"][:-1]]) == ". ".join(
            [x["content"] for x in example["rejected"][:-1]]
        )
        assert all(x["role"] != "system" for x in example["chosen"])

        prompt_messages = example["chosen"][:-1]
        chosen_messages = example["chosen"][-1:]
        rejected_messages = example["rejected"][-1:]

        processed_data["prompt"].append(
            tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        )
        processed_data["chosen"].append(
            tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        )
        processed_data["rejected"].append(
            tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        )

    dataset = Dataset.from_dict(processed_data)
    return dataset


def process_triplet_dataset(
    dataset: Dataset, tokenizer: PreTrainedTokenizer
) -> Dataset:
    def apply_chat_template(example):
        # Initialize variables to store the entire conversation
        conversation_history = []
        # Loop through all messages in the conversation
        for conv in example["conversation"]:
            if conv["from"] == "system":
                conversation_history.append({
                    "role": "system",
                    "content": conv["value"]
                })
            elif conv["from"] in ["human", "user"]:
                conversation_history.append({
                    "role": "user",
                    "content": conv["value"]
                })
            elif conv["from"] == "gpt":
                conversation_history.append({
                    "role": "assistant",
                    "content": conv["value"]
                })
        # Apply the chat template to the entire conversation history
        example["prompt"] = tokenizer.apply_chat_template(conversation_history, tokenize=False)
        # Add `chosen` and `rejected` with the eos token
        example["chosen"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": example["chosen"]['value']}], tokenize=False
        ) + tokenizer.eos_token
        example["rejected"] = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": example["rejected"]['value']}], tokenize=False
        ) + tokenizer.eos_token
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
            #print(f"One row from dataset subset '{dataset_name['subset']}':")
            #print(ds[0])
            dataset_list_validated.append(ds)

    dataset = concatenate_datasets(dataset_list_validated)
    dataset = dataset.train_test_split(test_size=0.01, shuffle=False)
    return dataset


def setup_special_tokens(
    model,
    tokenizer
):

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
    return model, tokenizer



def main(
    core_context: det.core.Context,
    training_args: TrainingArguments,
    det_callback: DetCallback,
    hparams: Dict[str, Any],
) -> None:
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    model_name= hparams["model_name"]

    if core_context.distributed.get_local_rank() == 0:
        snapshot_download(model_name, cache_dir=hf_cache_dir)
    _ = core_context.distributed.broadcast_local()  
    
    model = get_model(model_name,hf_cache_dir)

    tokenizer = get_tokenizer(
        model_name,
        truncation_side="left",
        padding_side="left",
        model_max_length=hparams['training_args']["max_length"],
        add_eos_token=False,
    )

    dataset =load_dpo_datasets(hparams["dataset"], tokenizer)
    #df = pd.DataFrame(dataset['train'][:])
    #df_sample = df.sample(n=20, random_state=42)
    #df_sample.to_csv('pfss/mlde/workspaces/mlde_wsp_P_AvemioAG/soumya_test/output/ORPO_train_dataset_Nemo_samples.csv',index=False)


    model, tokenizer = setup_special_tokens(
            model, tokenizer
        )

    
    trainer = ORPOTrainer(
        model=model,
        args= training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
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
        main(core_context, training_args, det_callback, hparams)
