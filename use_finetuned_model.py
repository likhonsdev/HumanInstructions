import os
import argparse
from typing import List, Dict, Any

# Import the Together API
from together import Together

# Import and load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, loading environment variables from OS")

def chat_with_model(
    model_name: str,
    api_key: str = None,
    system_prompt: str = "You are a helpful assistant that provides detailed and accurate responses.",
    use_mock: bool = True
):
    """
    Interactive chat with a fine-tuned model.
    
    Args:
        model_name: The name of the model to use
        api_key: Together API key (optional, defaults to environment variable)
        system_prompt: The system prompt to use
        use_mock: Whether to use mock mode for demonstration purposes
    """
    # Get the API key
    api_key = api_key or os.environ.get("TOGETHER_API_KEY") or "07f08ca73c50496a3406ff621912254a67370d576822f1921f77eed47e649545"
    if not api_key and not use_mock:
        print("Error: No API key provided. Set the TOGETHER_API_KEY environment variable or pass it as an argument.")
        return
    
    if use_mock:
        print("Using mock mode for demonstration purposes.")
        client = None
    else:
        print(f"Using API key: {api_key[:4]}...{api_key[-4:]}")
        # Initialize the Together client
        client = Together(api_key=api_key)

    print(f"\n===== Chatting with model: {model_name} =====")
    print("Type 'exit' to end the conversation.")
    print("Type 'new' to start a new conversation.\n")
    
    # Initialize the conversation
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        if user_input.lower() == 'new':
            print("Starting a new conversation.")
            conversation_history = []
            continue
        
        # Build the messages array
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history if any
        messages.extend(conversation_history)
        
        # Add the latest user message
        messages.append({"role": "user", "content": user_input})
        
        try:
            # Send the request to the model
            print("\nSending request to model...")
            
            if use_mock or client is None:
                # In mock mode, simulate a response based on the user's input
                print("(Using mock response)")
                
                # Generate a mock response based on the input
                if "hello" in user_input.lower() or "hi" in user_input.lower():
                    assistant_response = "Hello there! How can I help you today?"
                elif "what" in user_input.lower() and "name" in user_input.lower():
                    assistant_response = f"I'm a fine-tuned model based on {model_name}."
                elif "how" in user_input.lower() and "work" in user_input.lower():
                    assistant_response = "I was fine-tuned on a TruthfulQA dataset to provide truthful and accurate answers."
                elif "capital" in user_input.lower() and "france" in user_input.lower():
                    assistant_response = "The capital of France is Paris. It's known as the 'City of Light' and is famous for landmarks like the Eiffel Tower and the Louvre Museum."
                elif "truthful" in user_input.lower():
                    assistant_response = "I prioritize providing truthful information. I aim to give accurate answers based on facts."
                else:
                    assistant_response = "As a model fine-tuned on TruthfulQA dataset, I'm trained to provide accurate and truthful information. How can I help you with your question?"
            else:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=1024,
                        temperature=0.7
                    )
                    assistant_response = response.choices[0].message.content
                except Exception as e:
                    print(f"API request failed: {e}")
                    assistant_response = f"[Error: Could not get response from model. Please check your API key and model name.]"
            
            print(f"\nAssistant: {assistant_response}")
            
            # Add this exchange to the conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": assistant_response})
            
        except Exception as e:
            print(f"Error: {e}")

def single_prompt_response(
    model_name: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant that provides detailed and accurate responses.",
    api_key: str = None,
    use_mock: bool = True
):
    """
    Get a single response from the model for a given prompt.
    
    Args:
        model_name: The name of the model to use
        prompt: The prompt to send to the model
        system_prompt: The system prompt to use
        api_key: Together API key (optional, defaults to environment variable)
        use_mock: Whether to use mock mode for demonstration purposes
        
    Returns:
        The model's response
    """
    # Get the API key
    api_key = api_key or os.environ.get("TOGETHER_API_KEY") or "07f08ca73c50496a3406ff621912254a67370d576822f1921f77eed47e649545"
    
    # Set up the client or use mock mode
    if use_mock:
        print("Using mock mode for demonstration purposes.")
        client = None
    else:
        print(f"Using API key: {api_key[:4]}...{api_key[-4:]}")
        client = Together(api_key=api_key)
    
    # Set up the messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    print(f"\n===== Testing model: {model_name} =====")
    print(f"Prompt: {prompt}")
    print("Sending request to model...")
    
    # Get the response
    if use_mock or client is None:
        print("(Using mock response)")
        
        # Generate a mock response based on the input
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            response = "Hello there! How can I help you today?"
        elif "what" in prompt.lower() and "name" in prompt.lower():
            response = f"I'm a fine-tuned model based on {model_name}."
        elif "how" in prompt.lower() and "work" in prompt.lower():
            response = "I was fine-tuned on a TruthfulQA dataset to provide truthful and accurate answers."
        elif "capital" in prompt.lower() and "france" in prompt.lower():
            response = "The capital of France is Paris. It's known as the 'City of Light' and is famous for landmarks like the Eiffel Tower and the Louvre Museum."
        elif "truthful" in prompt.lower():
            response = "I prioritize providing truthful information. I aim to give accurate answers based on facts."
        else:
            response = "As a model fine-tuned on TruthfulQA dataset, I'm trained to provide accurate and truthful information. This is my response to your question."
    else:
        try:
            api_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.7
            )
            response = api_response.choices[0].message.content
        except Exception as e:
            print(f"API request failed: {e}")
            response = f"[Error: Could not get response from model. Please check your API key and model name.]"
    
    print(f"\nResponse: {response}")
    return response

def main():
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned model")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="The name of the fine-tuned model"
    )
    parser.add_argument(
        "--api_key", 
        type=str, 
        help="Together API key (can also be set via TOGETHER_API_KEY env var)"
    )
    parser.add_argument(
        "--system_prompt", 
        type=str, 
        default="You are a helpful assistant that provides detailed and accurate responses.",
        help="System prompt to use"
    )
    parser.add_argument(
        "--non_interactive",
        action="store_true",
        help="Run in non-interactive mode with a single prompt"
    )
    parser.add_argument(
        "--test_prompt",
        type=str,
        default="What is the capital of France?",
        help="Prompt to use in non-interactive mode"
    )
    
    args = parser.parse_args()
    
    if args.non_interactive:
        # Run in non-interactive mode with a single prompt
        single_prompt_response(
            model_name=args.model,
            prompt=args.test_prompt.strip('"\''),  # Remove quotes if present
            system_prompt=args.system_prompt.strip('"\''),  # Remove quotes if present
            api_key=args.api_key
        )
    else:
        # Run in interactive mode
        chat_with_model(
            model_name=args.model,
            api_key=args.api_key,
            system_prompt=args.system_prompt.strip('"\'')  # Remove quotes if present
        )

if __name__ == "__main__":
    main()
