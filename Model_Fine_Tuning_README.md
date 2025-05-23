# Model Fine-Tuning Toolkit for Together AI

This toolkit provides a comprehensive solution for fine-tuning language models using Together AI's platform. It's specifically designed to work with various datasets and provides all the necessary tools to prepare data, fine-tune models, monitor training, and evaluate results.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
- [Datasets](#datasets)
- [Fine-Tuning Options](#fine-tuning-options)
- [LoRA vs Full Fine-tuning](#lora-vs-full-fine-tuning)
- [Monitoring and Evaluation](#monitoring-and-evaluation)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

The Model Fine-Tuning Toolkit allows you to:

1. Prepare datasets in the required format for Together AI
2. Split data into training and validation sets
3. Upload data to Together AI
4. Create and monitor fine-tuning jobs
5. Evaluate fine-tuned models against base models
6. Use your fine-tuned models in interactive chat applications

## Installation

### Prerequisites

- Python 3.8+
- Together AI API key

### Setup

1. Install the required packages:

```bash
pip install -U together datasets transformers tqdm pandas python-dotenv requests
```

2. Clone this repository:

```bash
git clone https://github.com/yourusername/together-fine-tuning-toolkit.git
cd together-fine-tuning-toolkit
```

3. Set your Together AI API key:

```bash
# Option 1: Set as environment variable
export TOGETHER_API_KEY=your_api_key_here

# Option 2: Create a .env file
echo "TOGETHER_API_KEY=your_api_key_here" > .env
```

## Quick Start

### Example: Fine-tuning a model on TruthfulQA

```bash
# Fine-tune a model
python fine_tune_model.py --dataset truthful_qa --suffix my_truthful_model

# Use your fine-tuned model
python use_finetuned_model.py --model your-account/Meta-Llama-3.1-8B-Instruct-Reference-my_truthful_model-ft-xxxx
```

For a complete end-to-end example:

```bash
python example_usage.py
```

## Components

### 1. Fine-Tuning Script (`fine_tune_model.py`)

This is the main script for fine-tuning models, providing functionality for:

- Data preparation and formatting
- Data splitting (training/validation)
- Uploading data to Together AI
- Starting and monitoring fine-tuning jobs
- Evaluating model performance

#### Usage

```bash
python fine_tune_model.py --dataset [dataset_name] --suffix [model_suffix] [options]
```

#### Parameters

- `--dataset`: Dataset to use for fine-tuning (choices: drop, squad, toxicity, truthful_qa)
- `--model`: Base model to fine-tune (default: meta-llama/Meta-Llama-3.1-8B-Instruct-Reference)
- `--epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--batch_size`: Batch size (default: 'max')
- `--use_lora`: Whether to use LoRA fine-tuning (default: True)
- `--suffix`: Suffix to add to the model name (required)
- `--api_key`: Together API key (can also be set via environment variable)
- `--output_dir`: Directory to save output files (default: ./fine_tuning_output)

### 2. Interactive Chat Script (`use_finetuned_model.py`)

This script provides an interactive chat interface for your fine-tuned model.

#### Usage

```bash
python use_finetuned_model.py --model [model_name] [options]
```

#### Parameters

- `--model`: The name of the fine-tuned model (required)
- `--api_key`: Together API key (can also be set via environment variable)
- `--system_prompt`: System prompt to use (default: "You are a helpful assistant...")

### 3. Example Script (`example_usage.py`)

This script demonstrates the complete workflow for fine-tuning a model and using it.

#### Usage

```bash
python example_usage.py
```

## Datasets

The toolkit supports the following datasets:

1. **DROP** (Discrete Reasoning Over Paragraphs)
   - Question-answering dataset requiring discrete reasoning
   - Example: "How many yards did the Bears gain in the game?"

2. **SQuAD** (Stanford Question Answering Dataset)
   - Reading comprehension dataset
   - Example: "Who was the first woman astronaut?"

3. **TruthfulQA**
   - Dataset focused on generating truthful answers to questions
   - Designed to test a model's ability to avoid generating false information

4. **Toxicity Dataset**
   - Dataset for reducing toxic outputs
   - Fine-tuned models learn to generate non-toxic responses

## Fine-Tuning Options

### Learning Rate

The learning rate controls how quickly the model adapts to the new data. A higher learning rate means faster adaptation but may result in overshooting the optimal parameters.

```bash
python fine_tune_model.py --dataset squad --suffix squad_model --learning_rate 2e-5
```

### Number of Epochs

An epoch is one complete pass through the training dataset. More epochs generally lead to better performance, up to a point of diminishing returns.

```bash
python fine_tune_model.py --dataset drop --suffix drop_model --epochs 5
```

### Batch Size

The batch size determines how many examples the model processes before updating its parameters. The default 'max' optimizes for your hardware.

```bash
python fine_tune_model.py --dataset toxicity --suffix safe_model --batch_size 8
```

## LoRA vs Full Fine-tuning

### LoRA (Low-Rank Adaptation)

LoRA is a parameter-efficient fine-tuning technique that significantly reduces the number of trainable parameters:

- **Advantages**: Faster training, lower memory requirements, smaller model size
- **Best for**: Quick adaptation to new domains, limited compute resources

```bash
python fine_tune_model.py --dataset truthful_qa --suffix truthful_lora --use_lora
```

### Full Fine-tuning

Full fine-tuning updates all model parameters:

- **Advantages**: Potentially better performance, especially for complex tasks
- **Best for**: When you have substantial compute resources and time

```bash
python fine_tune_model.py --dataset squad --suffix squad_full --use_lora False
```

## Monitoring and Evaluation

### Monitoring Fine-tuning Jobs

You can monitor your fine-tuning job using the Together AI dashboard or API:

```python
from together import Together
client = Together(api_key="your_api_key")

# Check job status
job = client.fine_tuning.retrieve(id="ft-xxxx")
print(f"Status: {job.status}")

# Get job events/logs
events = client.fine_tuning.list_events(id="ft-xxxx")
for event in events.data:
    print(f"{event.created_at}: {event.message}")
```

### Evaluating Models

The toolkit provides functionality to evaluate your fine-tuned model against the base model:

```python
from fine_tune_model import evaluate_model

evaluate_model(
    base_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    finetuned_model="your-account/Model-Name-suffix-jobid",
    test_file="path/to/test/file.jsonl",
    n_samples=10
)
```

## Troubleshooting

### API Key Issues

If you're experiencing authentication errors:

1. Ensure your API key is correctly set in the environment or passed as an argument
2. Verify that the API key is active and has the necessary permissions

### Data Format Issues

If you encounter data formatting errors:

1. Check that your dataset files are in the correct format
2. Verify that the files are properly encoded (UTF-8)
3. Ensure the dataset has the expected columns

### Memory Issues

If you're running into memory errors:

1. Try reducing the batch size
2. Use LoRA instead of full fine-tuning
3. Choose a smaller base model

## API Reference

### Fine-tuning API

```python
# Start a fine-tuning job
response = client.fine_tuning.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
    training_file="file_xxxx",
    n_epochs=3,
    learning_rate=1e-5,
    batch_size="max",
    lora=True,
    suffix="my_suffix"
)

# Check job status
job = client.fine_tuning.retrieve(id="ft-xxxx")

# List job events
events = client.fine_tuning.list_events(id="ft-xxxx")
```

### Inference API

```python
# Use your fine-tuned model
response = client.chat.completions.create(
    model="your-account/Model-Name-suffix-jobid",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=128
)
print(response.choices[0].message.content)
```

---

For more information, please visit the [Together AI documentation](https://docs.together.ai/reference/fine-tuning-models).
