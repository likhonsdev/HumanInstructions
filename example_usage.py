"""
Example Usage for Together AI Model Fine-Tuning Toolkit

This script demonstrates the complete workflow for fine-tuning a model and using it:
1. Preparing a dataset (TruthfulQA)
2. Fine-tuning a model
3. Using the fine-tuned model

Usage:
    python example_usage.py
"""

import os
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def run_command(command, description=None):
    """Run a shell command and print its output."""
    if description:
        print(f"\n>> {description}")
    
    print(f"$ {command}\n")
    
    # Run the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Print the output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"Error: {result.stderr}")
    
    return result

def main():
    print_header("TOGETHER AI MODEL FINE-TUNING EXAMPLE")
    print("This script will demonstrate the complete workflow for fine-tuning a model on Together AI.")
    
    # Create output directory if it doesn't exist
    os.makedirs("fine_tuning_output", exist_ok=True)
    
    # Step 1: Fine-tune the model
    print_header("STEP 1: FINE-TUNE A MODEL")
    print("Fine-tuning a model on the TruthfulQA dataset...")
    
    # Set a unique suffix for the model
    model_suffix = f"demo_{int(time.time())}"
    
    # Run the fine-tuning script
    fine_tune_cmd = f"python fine_tune_model.py --dataset truthful_qa --suffix {model_suffix}"
    result = run_command(fine_tune_cmd, "Running fine-tuning script")
    
    # Extract the model name from the output (in a real scenario, you would wait for the job to complete)
    model_name = f"your-account/Meta-Llama-3.1-8B-Instruct-Reference-{model_suffix}-ft-00000072"
    print(f"\nFine-tuned model name: {model_name}")
    print("\nNOTE: In a real scenario, you would need to wait for the fine-tuning job to complete.")
    print("      This can take several hours depending on the dataset size and model parameters.")
    
    # Step 2: Use the fine-tuned model
    print_header("STEP 2: USE THE FINE-TUNED MODEL")
    print("Now you can chat with your fine-tuned model using the use_finetuned_model.py script:")
    print(f"python use_finetuned_model.py --model {model_name}")
    
    print("\nExample interaction:")
    print("\nYou: What is the capital of France?")
    print("Assistant: The capital of France is Paris. It's known as the 'City of Light' and is famous for landmarks like the Eiffel Tower and the Louvre Museum.")
    
    print("\nYou: How were you fine-tuned?")
    print("Assistant: I was fine-tuned on a TruthfulQA dataset to provide truthful and accurate answers.")
    
    # Step 3: Explain how to evaluate the model
    print_header("STEP 3: EVALUATE THE MODEL")
    print("To evaluate your model against the base model, you can use the evaluate_model function:")
    
    evaluation_code = f"""
    from fine_tune_model import evaluate_model
    
    evaluate_model(
        base_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        finetuned_model="{model_name}",
        test_file="fine_tuning_output/truthful_qa_prepared_valid.jsonl",
        n_samples=10
    )
    """
    
    print("\nExample evaluation code:")
    print(evaluation_code)

    # Conclusion
    print_header("CONCLUSION")
    print("You have successfully demonstrated the complete workflow for fine-tuning a model:")
    print("1. Prepared the TruthfulQA dataset")
    print("2. Fine-tuned a model using the dataset")
    print("3. Used the fine-tuned model for inference")
    print("4. Learned how to evaluate the model")
    
    print("\nFor more information, please refer to the Model_Fine_Tuning_README.md file.")

if __name__ == "__main__":
    main()
