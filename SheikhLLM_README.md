# SheikhLLM: Custom Fine-Tuned LLM

This README provides instructions for fine-tuning and using your custom SheikhLLM model.

## Overview

SheikhLLM is your personalized language model, fine-tuned on selected datasets using the Together AI platform. The model is designed to enhance the capabilities of a base LLM with your specific requirements and use cases.

## Requirements

- Python 3.8+
- Together AI API key (already configured in your .env file)
- Required Python packages (installed with your existing environment)

## Quick Start

### 1. Fine-Tune SheikhLLM

To start the fine-tuning process, run:

```bash
python fine_tune_sheikhllm.py
```

By default, the script runs in mock mode, which doesn't make actual API calls to Together AI. To use real API calls for actual model fine-tuning, use:

```bash
python fine_tune_sheikhllm.py --use_real_api
```

This requires a valid Together AI API key in your .env file.

This will:
- Use the default TruthfulQA dataset
- Use the Meta-Llama-3.1-8B-Instruct-Reference as the base model
- Apply LoRA fine-tuning (parameter-efficient)
- Run for 3 epochs with a learning rate of 1e-5

### 2. Custom Fine-Tuning Options

You can customize the fine-tuning process with various options:

```bash
python fine_tune_sheikhllm.py --dataset squad --epochs 5 --learning_rate 2e-5
```

Available datasets:
- `drop` - Discrete Reasoning Over Paragraphs (complex QA)
- `squad` - Stanford Question Answering Dataset (reading comprehension)
- `toxicity` - Toxicity reduction dataset (safer outputs)
- `truthful_qa` - Dataset focused on truthful answers (default)
- `tldr` - Summarization dataset

### 3. Using SheikhLLM

Once your model has completed fine-tuning, you can chat with it using:

```bash
python use_sheikhllm.py --model_id <fine-tuning-job-id>
```

Example:
```bash
python use_sheikhllm.py --model_id ft-abcd1234
```

If you don't provide a model ID, the script will prompt you to enter one or use mock mode for testing.

## Fine-Tuning Process

### Step 1: Data Preparation

The fine-tuning script will:
1. Load the selected dataset
2. Convert it to the required format for Together AI
3. Split the data into training and validation sets
4. Validate the data format

### Step 2: Model Fine-Tuning

After data preparation, the script will:
1. Upload the prepared data to Together AI
2. Configure the fine-tuning job with your settings
3. Submit the job to the Together AI platform
4. Provide you with a job ID for tracking

### Step 3: Monitoring (Manual)

You can monitor your fine-tuning job:
1. Visit the Together AI dashboard
2. Check the job status using the provided job ID
3. Wait for the job to complete (this can take several hours)

### Step 4: Using the Model

Once fine-tuning is complete, you can:
1. Chat interactively with your model using the provided script
2. Integrate the model into your applications using the Together AI API

## GitHub Actions Integration

You can use GitHub Actions to automate the fine-tuning process, which is particularly useful for periodic retraining or team workflows.

### Using the GitHub Actions Workflow

The included `.github/workflows/sheikhllm_runner.yml` file provides automation for:
- Scheduled weekly fine-tuning (Monday 1:00 AM UTC) using real Together API calls
- On-demand fine-tuning with customizable parameters
- Automatic validation and artifact storage

**Note:** The GitHub Actions workflow always uses `--use_real_api` to make actual API calls to Together AI, unlike local runs, which use mock mode by default.

To run the workflow manually:
1. Go to your GitHub repository's "Actions" tab
2. Select "SheikhLLM Runner" from the workflows list
3. Click "Run workflow"
4. Select your desired dataset, epochs, and learning rate
5. Click "Run workflow" to start the process

### Workflow Configuration

The workflow requires:
- A GitHub secret named `TOGETHER_API_KEY` with your Together AI API key
- Proper permissions to run GitHub Actions in your repository

### Monitoring Workflow Runs

1. Check the "Actions" tab in your GitHub repository
2. Select the specific workflow run
3. Review logs for progress and results
4. Download training artifacts from successful runs

## Advanced Usage

### LoRA vs Full Fine-Tuning

By default, SheikhLLM uses LoRA (Low-Rank Adaptation) for efficient fine-tuning:

- **LoRA** (default): Faster, uses less memory, smaller model size
- **Full Fine-Tuning**: May provide better performance for complex tasks

To use full fine-tuning:

```bash
python fine_tune_sheikhllm.py --use_lora False
```

### Customizing the System Prompt

When using your model, you can customize its behavior with a system prompt:

```bash
python use_sheikhllm.py --model_id ft-abcd1234 --system_prompt "You are SheikhLLM, an expert in Islamic finance and Shariah-compliant solutions."
```

## Troubleshooting

### Fine-Tuning Issues

- **Authentication Errors**: Check your Together API key in .env
- **Data Format Errors**: Verify dataset file format and encoding
- **Quota Limitations**: Check your Together AI account limits

### Usage Issues

- **Model Not Found**: Ensure fine-tuning job is complete and ID is correct
- **Rate Limit Errors**: Slow down request rate or check account limits
- **Response Quality**: Try adjusting the system prompt or temperature

## Model Files

The fine-tuned model will be hosted on Together AI's platform with a name like:

```
your-account/Meta-Llama-3.1-8B-Instruct-Reference-sheikhllm-ft-xxxxxxxx
```

You don't need to download any model files - the model is accessed via API calls.

---

For more details about the fine-tuning toolkit, refer to the main `Model_Fine_Tuning_README.md` file.
