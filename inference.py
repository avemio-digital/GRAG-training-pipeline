import argparse
import glob
from typing import Optional
import os
import pandas as pd
from datasets import load_dataset
from determined.experimental import client

from chat_format import maybe_add_generation_prompt
from utils import get_model, get_tokenizer

os.environ["HF_TOKEN"] = "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"

def main(
    exp_id: Optional[int],
    trial_id: Optional[int],
    device: str,
    output_file: Optional[str],
    number_of_samples: int,
) -> None:
    model_name = "/home/siavash/determined-examples/blog/llm-finetuning-3/checkpoint-5000"

    if exp_id is None and trial_id is None:
        checkpoint_dir = model_name
        is_base_model = True
    else:
        exp = client.get_experiment(exp_id) if exp_id else client.get_trial(trial_id)
        checkpoint = exp.list_checkpoints(
            max_results=1,
            sort_by=client.CheckpointSortBy.BATCH_NUMBER,
            order_by=client.OrderBy.DESCENDING,
        )[0]
        checkpoint_dir = checkpoint.download(mode=client.DownloadMode.MASTER)
        checkpoint_dir = glob.glob(f"{checkpoint_dir}/checkpoint-*")[0]
        is_base_model = False

    model = get_model(checkpoint_dir, inference=True, device_map=device)
    tokenizer = get_tokenizer(
        checkpoint_dir,
        truncation_side="right",
        model_max_length=8192,
        add_eos_token=False,
    )
    results = []
    dataset = load_dataset(
        "avemio-digital/inference_determined_alpaca", split="train[:20]"
    )
    for element in dataset:

     #   print(element['conversations'])
        if not is_base_model:
            formatted = tokenizer.apply_chat_template(
                conversation=[
                    {
                        "role": "user",
                        "content": str(element["system"]) + "\n" + str(element["conversations"]),
                    },
                ],
                tokenize=False,
            )
            formatted = maybe_add_generation_prompt(formatted)
        else:
            formatted = str(element["Instruction"]) + "\n" + str(element["System"])

        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        input_str = tokenizer.batch_decode(inputs["input_ids"])[0]
      #  print(f"Model input: {input_str}")

        outputs = model.generate(
            **inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=50,do_sample=True
 
        )
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.batch_decode(
            outputs[:, input_length:], skip_special_tokens=True
        )
      #  print(f"\n\nLLM response:\n{response[0]}")

        element_result = {
            "input": input_str,
            "output": response[0]
        }
        results.append(element_result)
        # Append the current result to the results list.
      #  results["input"].append(input_str)
      #  results["output"].append(response[0])

    if output_file:
        print(results)
        df = pd.DataFrame.from_dict(results)
        df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=None, required=False)
    parser.add_argument("--trial_id", type=int, default=None, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default=None, required=True)
    parser.add_argument("--number_of_samples", type=int, default=100, required=False)
    args = parser.parse_args()

    main(
        args.exp_id,
        args.trial_id,
        args.device,
        args.output_file,
        args.number_of_samples,
    )
