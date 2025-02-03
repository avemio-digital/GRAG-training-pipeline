<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">GRAG-HESSIANAI-TRAINING-PIPELINE</h1></p>
<p align="center">
	<em>Empowering AI with Determined Precision and Speed</em>
</p>
<p align="center">
	<!-- local repository, no metadata badges. --></p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=default&logo=GNU-Bash&logoColor=white" alt="GNU%20Bash">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

The GRAG-HessianAI-Training-Pipeline project streamlines deep learning model training by orchestrating dataset processing, hyperparameter tuning, and distributed computing. It offers seamless integration with external libraries and custom utilities, optimizing model training efficiency. Targeting AI researchers and developers, it simplifies the training workflow for enhanced model performance and evaluation.

---

##  Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Utilizes **DeepSpeed** for optimization and gradient checkpointing</li><li>Integrates with external libraries and custom utilities for streamlined workflow</li><li>Defines container bind mounts and environment configurations in `config.yaml`</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Code files maintain clear structure and readability</li><li>Utilizes **Hugging Face Transformers** for model management</li><li>Implements distributed computing and fine-tuning techniques</li></ul> |
| 📄 | **Documentation** | <ul><li>Comprehensive documentation in **Python** with various file formats</li><li>Facilitates dependencies management with `requirements.txt`</li><li>Provides detailed training pipeline configurations in YAML files</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates with **Determined AI** for distributed training</li><li>Utilizes **Transformers, Datasets, and scikit-learn** for functionality</li><li>Facilitates model retrieval and tokenization with custom utilities</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Codebase structured into modular components for easy maintenance</li><li>Separates training pipeline configurations into individual files</li><li>Enables efficient model deployment and maintenance within the architecture</li></ul> |
| 🧪 | **Testing**       | <ul><li>Includes testing commands using **pytest** for codebase validation</li><li>Ensures model training and evaluation functionality is tested</li><li>Facilitates efficient model training and evaluation within the project architecture</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimizes training pipeline with **mixed precision** and **optimizer settings**</li><li>Configures **gradient accumulation** and **zero optimization** for improved performance</li><li>Fine-tunes batch sizes and clipping for enhanced training efficiency</li></ul> |
| 🛡️ | **Security**      | <ul><li>Ensures secure model deployment and real-time inference tasks</li><li>Handles input processing and result storage securely</li><li>Upgrades dependencies and fixes bugs for improved security</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Manages project dependencies using **pip** and `requirements.txt`</li><li>Specifies required packages and versions for seamless integration</li><li>Facilitates efficient model training and evaluation within the architecture</li></ul> |

---

##  Project Structure

```sh
└── GRAG-HessianAI-Training-Pipeline/
    └── GRAG_Hessian_AI_Determined_Training_Pipeline
        ├── Orpo_attendee.yaml
        ├── README.md
        ├── chat_format.py
        ├── config.yaml
        ├── cpt.yaml
        ├── cpt_finetune.py
        ├── cptold.txt
        ├── dpo.yaml
        ├── dpo_finetune.py
        ├── ds_configs
        ├── inference.py
        ├── lora.yaml
        ├── lora_finetune.py
        ├── lora_utils.py
        ├── metadata.json
        ├── old_startup-hook.sh
        ├── orpo.yaml
        ├── orpo_finetune.py
        ├── requirements.txt
        ├── sft.yaml
        ├── sft_finetune.py
        ├── startup-hook.sh
        ├── untitled.txt
        ├── utils.py
        └── utils_lora_old.py
```


###  Project Index
<details open>
	<summary><b><code>GRAG-HESSIANAI-TRAINING-PIPELINE/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			</table>
		</blockquote>
	</details>
	<details> <!-- GRAG_Hessian_AI_Determined_Training_Pipeline Submodule -->
		<summary><b>GRAG_Hessian_AI_Determined_Training_Pipeline</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/Orpo_attendee.yaml'>Orpo_attendee.yaml</a></b></td>
				<td>- Defines training pipeline configuration for LLAMA_8B_ORPO_attendee, specifying resources, hyperparameters, and environment settings<br>- Sets up deep learning model training with specific dataset subsets and training arguments, including batch size, learning rate, and evaluation strategy<br>- Configures deepspeed for optimization and gradient checkpointing.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/cpt_finetune.py'>cpt_finetune.py</a></b></td>
				<td>- The code file `cpt_finetune.py` orchestrates the training pipeline by loading datasets, setting up special tokens, and initializing the training process<br>- It leverages distributed computing and fine-tuning techniques to train a model based on specified hyperparameters<br>- The file integrates with external libraries and custom utilities to streamline the training workflow within the project architecture.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/requirements.txt'>requirements.txt</a></b></td>
				<td>- Facilitates project dependencies management by specifying required packages and versions<br>- This file ensures the project can leverage essential libraries like transformers, datasets, and scikit-learn for seamless integration and functionality within the codebase architecture.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/lora_finetune.py'>lora_finetune.py</a></b></td>
				<td>- The code file orchestrates the loading and processing of datasets for training a conversational AI model<br>- It ensures the datasets are in the correct format and applies necessary transformations<br>- Additionally, it sets up special tokens for the model and initiates the training process with the specified training arguments and callbacks.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/sft_finetune.py'>sft_finetune.py</a></b></td>
				<td>- The code file orchestrates the training pipeline for fine-tuning a language model using a self-feeding chat dataset<br>- It loads the dataset, sets up special tokens, formats prompts, and initiates training with specific configurations<br>- The file integrates with external libraries and tools to facilitate efficient model training and evaluation within the project architecture.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/chat_format.py'>chat_format.py</a></b></td>
				<td>- Generate chat ML templates for user, system, and assistant messages based on predefined roles within the chat messages<br>- The code defines templates for different message roles and formats them accordingly for ML processing<br>- Additionally, it provides functions to retrieve assistant prompts and template IDs for responses, enhancing the chat generation process.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/metadata.json'>metadata.json</a></b></td>
				<td>- Tracks the progress and identification of a specific trial within the training pipeline, capturing the number of completed steps and the unique trial ID<br>- This metadata file plays a crucial role in monitoring and managing the training process within the project architecture.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/config.yaml'>config.yaml</a></b></td>
				<td>Define container bind mounts, environment configurations, and resource allocations for the training pipeline in the project architecture.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/cptold.txt'>cptold.txt</a></b></td>
				<td>- Define training pipeline configuration for Qwen1.5 model with specific hyperparameters and resources<br>- Specifies dataset subsets, model details, and training settings for the AI model.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/sft.yaml'>sft.yaml</a></b></td>
				<td>- Facilitates training a deep learning model with Determined AI, leveraging a specific dataset and hyperparameters configuration<br>- Manages resource allocation, environment setup, and training parameters for the training pipeline.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/untitled.txt'>untitled.txt</a></b></td>
				<td>Patch the HF callback script to handle additional metric types for improved training pipeline functionality.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/lora.yaml'>lora.yaml</a></b></td>
				<td>- Facilitates training a language model using a specific dataset and hyperparameters<br>- Manages resource allocation, environment setup, and training configuration for the Nemo_12B_Lora_ORPO_attendee project.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/utils.py'>utils.py</a></b></td>
				<td>- Facilitates model retrieval and tokenization for the AI training pipeline<br>- Handles model loading based on inference mode, customizes tokenization parameters, and downloads model checkpoints<br>- Integrates with Determined for distributed training.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/dpo.yaml'>dpo.yaml</a></b></td>
				<td>- Facilitates training pipeline configuration for deep learning model fine-tuning with Determined AI<br>- Specifies resources, hyperparameters, and environment settings for the training job<br>- Manages data subsets, loss function, and training strategies<br>- Enables efficient model training and evaluation.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/dpo_finetune.py'>dpo_finetune.py</a></b></td>
				<td>- The code file orchestrates the training pipeline for a Determined AI model by loading datasets, processing conversation formats, and training a model using specified hyperparameters<br>- It ensures data compatibility, tokenization, and model training with distributed support, ultimately facilitating efficient model training and evaluation.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/cpt.yaml'>cpt.yaml</a></b></td>
				<td>- Facilitates training of a custom AI model using DeepSpeed with specific hyperparameters and configurations<br>- Manages data subsets, batch sizes, mixed precision, and gradient accumulation steps for efficient training<br>- Enables fine-tuning of the Mistral-Nemo-Base-2407 model on the GRAG-CPT-Hessian-AI dataset for various language tasks.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/utils_lora_old.py'>utils_lora_old.py</a></b></td>
				<td>- Facilitates model retrieval, tokenizer setup, and checkpoint downloading for the AI training pipeline<br>- Handles model variations, including Lora integration, and ensures proper tokenization<br>- Enables seamless model deployment and maintenance within the project architecture.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/inference.py'>inference.py</a></b></td>
				<td>- The code in `inference.py` orchestrates model inference using Determined AI, generating responses based on input data<br>- It leverages a pre-trained model to process conversations and produce corresponding outputs<br>- The script facilitates the deployment of the model for real-time inference tasks, handling input processing, model generation, and result storage.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/orpo_finetune.py'>orpo_finetune.py</a></b></td>
				<td>- The code file orchestrates the training pipeline for fine-tuning a conversational AI model using the ORPO technique<br>- It handles dataset processing, model setup, and training execution<br>- The code integrates with Determined AI for distributed training and leverages Hugging Face Transformers for model management<br>- The main function initiates the training process based on specified parameters and hyperparameters.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/lora_utils.py'>lora_utils.py</a></b></td>
				<td>- Enables retrieval of pre-trained language models and tokenizers, facilitating model inference and training<br>- Supports custom configurations for model architecture and tokenization<br>- Additionally, provides functionality for downloading model checkpoints and defining tokenization functions.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/startup-hook.sh'>startup-hook.sh</a></b></td>
				<td>Patch startup script to upgrade dependencies and fix a bug in the Hugging Face callback module for the AI training pipeline.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/orpo.yaml'>orpo.yaml</a></b></td>
				<td>- Facilitates training of a mini ORPO model with attendee-specific data subsets<br>- Utilizes a specific deep learning model and dataset for training, with customized training arguments and hyperparameters<br>- Implements a deepspeed configuration for efficient training.</td>
			</tr>
			<tr>
				<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/old_startup-hook.sh'>old_startup-hook.sh</a></b></td>
				<td>Patch the startup script to upgrade dependencies and modify a specific condition for training metrics handling.</td>
			</tr>
			</table>
			<details>
				<summary><b>ds_configs</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/ds_configs/ds_config_stage_1.json'>ds_config_stage_1.json</a></b></td>
						<td>- Define training pipeline configurations for stage 1 with automatic settings for mixed precision, optimizer, scheduler, zero optimization, gradient accumulation, and gradient clipping<br>- Includes options for batch sizes and FLOPs profiling.</td>
					</tr>
					<tr>
						<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/ds_configs/ds_config_stage_2_cpu_offload.json'>ds_config_stage_2_cpu_offload.json</a></b></td>
						<td>- Define CPU offload configuration for stage 2 training pipeline in ds_config_stage_2_cpu_offload.json<br>- Configure FP16, AdamW optimizer, WarmupLR scheduler, and zero optimization settings for gradient accumulation and clipping<br>- Fine-tune training batch size and micro-batch size per GPU<br>- Optionally enable FLOPs profiler for detailed performance analysis.</td>
					</tr>
					<tr>
						<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/ds_configs/ds_config_stage_2.json'>ds_config_stage_2.json</a></b></td>
						<td>- Define configuration settings for stage 2 of the training pipeline, specifying optimization parameters, gradient accumulation, and zero optimization strategies<br>- This file plays a crucial role in fine-tuning training performance and resource utilization within the project architecture.</td>
					</tr>
					<tr>
						<td><b><a href='/Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline/blob/master/GRAG_Hessian_AI_Determined_Training_Pipeline/ds_configs/ds_config_stage_3.json'>ds_config_stage_3.json</a></b></td>
						<td>- Optimizes training pipeline by configuring mixed precision, optimizer settings, and zero optimization parameters for efficient model training<br>- Fine-tunes batch sizes, gradient accumulation, and clipping for improved performance.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with GRAG-HessianAI-Training-Pipeline, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


###  Installation

Install GRAG-HessianAI-Training-Pipeline using one of the following methods:

**Build from source:**

1. Clone the GRAG-HessianAI-Training-Pipeline repository:
```sh
❯ git clone ../GRAG-HessianAI-Training-Pipeline
```

2. Navigate to the project directory:
```sh
❯ cd GRAG-HessianAI-Training-Pipeline
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r GRAG_Hessian_AI_Determined_Training_Pipeline/requirements.txt
```




###  Usage
Run GRAG-HessianAI-Training-Pipeline using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python {entrypoint}
```


###  Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pytest
```


---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

##  Contributing

- **💬 [Join the Discussions](https://LOCAL/Downloads/GRAG-HessianAI-Training-Pipeline/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://LOCAL/Downloads/GRAG-HessianAI-Training-Pipeline/issues)**: Submit bugs found or log feature requests for the `GRAG-HessianAI-Training-Pipeline` project.
- **💡 [Submit Pull Requests](https://LOCAL/Downloads/GRAG-HessianAI-Training-Pipeline/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your LOCAL account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone /Users/soumyapaul/Downloads/GRAG-HessianAI-Training-Pipeline
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to LOCAL**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://LOCAL{/Downloads/GRAG-HessianAI-Training-Pipeline/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Downloads/GRAG-HessianAI-Training-Pipeline">
   </a>
</p>
</details>

---

##  License

This project is protected under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) License.

---

##  Acknowledgments

--- Contributors: Marcel Rosiak, Soumya Paul, Siavash Mollaebrahim, Zain Ul Haq


