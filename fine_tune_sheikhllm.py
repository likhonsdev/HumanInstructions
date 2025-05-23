"""
Fine-tune a custom SheikhLLM model using the Together AI platform.

This script sets up and launches a fine-tuning job that will create a model with 
the suffix "sheikhllm", effectively creating your custom SheikhLLM model.
"""

import os
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Fine-tune SheikhLLM model')
    parser.add_argument('--dataset', type=str, default='truthful_qa', 
                        choices=['drop', 'squad', 'toxicity', 'truthful_qa', 'tldr'],
                        help='Dataset to use for fine-tuning')
    parser.add_argument('--base_model', type=str, 
                        default='meta-llama/Meta-Llama-3.1-8B-Instruct-Reference',
                        help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help='Learning rate for fine-tuning')
    parser.add_argument('--use_lora', action='store_true', default=True,
                        help='Whether to use LoRA fine-tuning (recommended)')
    parser.add_argument('--use_real_api', action='store_true', default=False,
                        help='Use real Together API calls instead of mock mode')
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs("fine_tuning_output", exist_ok=True)
    
    # Set up the command
    cmd = [
        "python", "fine_tune_model.py",
        "--dataset", args.dataset,
        "--model", args.base_model,
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
        "--suffix", "sheikhllm",  # This will create the sheikhllm model
        "--output_dir", "./fine_tuning_output"
    ]
    
    if args.use_lora:
        cmd.append("--use_lora")
    
    # Add flag for real API usage 
    if args.use_real_api:
        cmd.append("--use_real_api")
    
    # Print details
    print("\n===== FINE-TUNING SHEIKHLLM MODEL =====")
    print(f"Dataset: {args.dataset}")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Using LoRA: {args.use_lora}")
    print("\nStarting fine-tuning process...\n")
    
    # Run the command
    process = subprocess.run(" ".join(cmd), shell=True)
    
    if process.returncode == 0:
        print("\n===== FINE-TUNING JOB SUBMITTED SUCCESSFULLY =====")
        print("Your SheikhLLM model is now being fine-tuned!")
        print("\nThe fine-tuned model will be named:")
        print(f"your-account/{args.base_model.split('/')[-1]}-sheikhllm-ft-xxxxxxxx")
        print("\nOnce the fine-tuning job is complete, you can use your model with:")
        print("python use_finetuned_model.py --model your-model-name")
    else:
        print("\n===== FINE-TUNING JOB SUBMISSION FAILED =====")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()
