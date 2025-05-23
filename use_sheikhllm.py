"""
Interactive chat interface for the SheikhLLM model.

This script provides a user-friendly way to chat with your fine-tuned SheikhLLM model
after the fine-tuning process is complete.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Chat with the SheikhLLM model')
    parser.add_argument('--model_id', type=str, required=False,
                        help='The job ID of your fine-tuned model (e.g., ft-xxxxxxxx)')
    parser.add_argument('--base_model', type=str, 
                        default='meta-llama/Meta-Llama-3.1-8B-Instruct-Reference',
                        help='Base model that was fine-tuned')
    parser.add_argument('--system_prompt', type=str, 
                        default="You are SheikhLLM, a helpful AI assistant fine-tuned to provide accurate and truthful information.",
                        help='System prompt to set the behavior of the model')
    parser.add_argument('--test_mode', type=bool, default=False,
                        help='Run in test mode (single prompt and exit)')
    parser.add_argument('--prompt', type=str, default="What is the capital of France?",
                        help='Test prompt to use in test mode')
    args = parser.parse_args()
    
    # Construct the model name
    if args.model_id:
        model_name = f"your-account/{args.base_model.split('/')[-1]}-sheikhllm-{args.model_id}"
    else:
        # If no model ID provided, ask the user to input it
        print("\n===== WELCOME TO SHEIKHLLM CHAT =====")
        print("To chat with your SheikhLLM model, we need the fine-tuning job ID.")
        print("Example: If your model is 'your-account/Meta-Llama-3.1-8B-sheikhllm-ft-abcd1234',")
        print("         then the job ID is 'ft-abcd1234'.")
        print("\nIf you've just started the fine-tuning process, the job may not be complete yet.")
        print("You can check the status using the Together AI dashboard.")
        
        model_id = input("\nPlease enter your fine-tuning job ID (or leave empty to use mock mode): ")
        
        if model_id:
            model_name = f"your-account/{args.base_model.split('/')[-1]}-sheikhllm-{model_id}"
        else:
            # Use mock mode with a placeholder model name
            model_name = f"your-account/{args.base_model.split('/')[-1]}-sheikhllm-ft-mock"
            print(f"\nUsing mock mode with model: {model_name}")
    
    # Configure command based on mode
    if args.test_mode:
        print("\n===== TESTING SHEIKHLLM MODEL =====")
        print(f"Model: {model_name}")
        print(f"Test prompt: {args.prompt}")
        print("\nExecuting test...\n")
        
        # Create job_info.txt with the model ID if it doesn't exist
        # This is useful for GitHub Actions to track the model ID
        if args.model_id:
            job_info_dir = Path("fine_tuning_output")
            job_info_dir.mkdir(exist_ok=True)
            with open(job_info_dir / "job_info.txt", "w") as f:
                f.write(f"Job ID: {args.model_id}\n")
                f.write(f"Model: {model_name}\n")
        
        # Run a single test prompt and exit
        cmd = [
            "python", "use_finetuned_model.py",
            "--model", model_name,
            "--system_prompt", f'"{args.system_prompt}"',
            "--test_prompt", f'"{args.prompt}"',
            "--non_interactive"
        ]
        
        result = subprocess.run(" ".join(cmd), shell=True)
        sys.exit(result.returncode)
    else:
        # Interactive mode
        print("\n===== STARTING SHEIKHLLM CHAT =====")
        print(f"Model: {model_name}")
        print("Type 'exit' to end the conversation.")
        print("Type 'new' to start a new conversation.")
        print("\nInitializing chat...\n")
        
        # Start the chat using the use_finetuned_model.py script
        cmd = [
            "python", "use_finetuned_model.py",
            "--model", model_name,
            "--system_prompt", f'"{args.system_prompt}"'
        ]
        
        # Run the command
        subprocess.run(" ".join(cmd), shell=True)

if __name__ == "__main__":
    main()
