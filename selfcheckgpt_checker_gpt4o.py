import os
import json
import time
import argparse
import sys
from typing import List, Dict, Any
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

CAPTIONS_PATH = "llava_captions.json"
SELFCHECK_RESULTS_PATH = "selfcheck_results.json"

TOKEN_LIMIT = 800000  # 800k token limit per day for GPT-4o
total_tokens_used = 0

def create_checker_prompt(context: str, sentence: str):
    """Create a prompt for checking hallucinations using the specified format"""
    return f"Context: {context}\nSentence: {sentence}\nIs the sentence supported by the context above? Answer Yes or No:"

def check_with_gpt4o(client, first_caption: str, sample_captions: List[str], model="gpt-4o"):
    """Check if the first caption is supported by the sample captions"""
    global total_tokens_used
    
    consistency_results = []
    for sample in sample_captions:
        try:
            prompt = create_checker_prompt(sample, first_caption)
            print(f"Prompt: {prompt}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=5  # We only need a short response (Yes/No)
            )
            tokens_in_request = response.usage.total_tokens
            total_tokens_used += tokens_in_request
            print(f"  Tokens used: {tokens_in_request} | Total used: {total_tokens_used} | Remaining: {TOKEN_LIMIT - total_tokens_used}")
            response_text = response.choices[0].message.content.strip().lower()
            if 'yes' in response_text:
                consistency_results.append(True)
            elif 'no' in response_text:
                consistency_results.append(False)
            else:
                print(f"Ambiguous response from GPT-4o: {response_text}")
                consistency_results.append(False)
                
        except Exception as e:
            print(f"Error querying GPT-4o: {str(e)}")
            consistency_results.append(False)
            time.sleep(5)
    consistency = sum(consistency_results) / len(consistency_results) if consistency_results else 0
    
    return {
        "consistency": consistency,
        "is_consistent": consistency >= 0.5,  # Using 0.5 as threshold
        "supported_count": sum(consistency_results),
        "total_samples": len(consistency_results),
        "consistency_results": consistency_results
    }

def save_progress_state(results, last_processed_index, current_processing=None):
    """
    Save the current progress state to a file
    
    Args:
        results: List of processed results
        last_processed_index: Index of the last fully processed image
        current_processing: ID of image currently being processed (if any)
    """
    progress_path = f"{SELFCHECK_RESULTS_PATH}.progress"
    progress_state = {
        "last_processed_index": last_processed_index,
        "total_tokens_used": total_tokens_used,
        "current_processing": current_processing
    }
    
    with open(progress_path, 'w') as f:
        json.dump(progress_state, f)
    with open(SELFCHECK_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Progress saved. Last processed index: {last_processed_index}, Total tokens: {total_tokens_used}")

def load_progress_state():
    """Load the previous progress state if it exists"""
    global total_tokens_used
    
    progress_path = f"{SELFCHECK_RESULTS_PATH}.progress"
    current_processing = None
    
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress_state = json.load(f)
            total_tokens_used = progress_state.get("total_tokens_used", 0)
            current_processing = progress_state.get("current_processing", None)
            return (
                progress_state.get("last_processed_index", -1), 
                progress_state.get("total_tokens_used", 0),
                current_processing
            )
    return -1, 0, None

def main():
    parser = argparse.ArgumentParser(description="Check captions for hallucinations using SelfCheckGPT approach")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for consistency")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    parser.add_argument("--token_limit", type=int, default=None, help="Token limit for the session")
    args = parser.parse_args()
    global TOKEN_LIMIT
    if args.token_limit is not None:
        TOKEN_LIMIT = args.token_limit
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with open(CAPTIONS_PATH, 'r') as f:
        captions_data = json.load(f)
    captions_by_id = {item["id"]: item for item in captions_data}
    results = []
    if os.path.exists(SELFCHECK_RESULTS_PATH):
        with open(SELFCHECK_RESULTS_PATH, 'r') as f:
            results = json.load(f)
    processed_images = {result["id"]: True for result in results}
    last_processed_index, loaded_tokens, current_processing = load_progress_state()
    print(f"Starting with {total_tokens_used} tokens already used. Limit: {TOKEN_LIMIT}")
    if current_processing is not None and current_processing in captions_by_id:
        print(f"Resuming interrupted processing of image {current_processing}")
        results = [r for r in results if r["id"] != current_processing]
        processed_images.pop(current_processing, None)
    if args.max_images is not None:
        captions_data = captions_data[:args.max_images]
    total_images = len(captions_data)
    for i, data in enumerate(captions_data):
        if i <= last_processed_index and data["id"] != current_processing:
            print(f"Skipping index {i} (already processed in previous run)")
            continue
        
        image_id = data["id"]
        if image_id in processed_images:
            print(f"Skipping {image_id} - already in results")
            continue
        
        print(f"\nProcessing {i+1}/{total_images}: {image_id}")
        save_progress_state(results, last_processed_index, image_id)
        if total_tokens_used >= TOKEN_LIMIT:
            print(f"Token limit of {TOKEN_LIMIT} reached. Saving progress and exiting.")
            save_progress_state(results, last_processed_index, None)
            sys.exit(0)
        
        first_caption = data["first_caption"]
        samples = data["samples"]
        ground_truth = data["ground_truth"]
        consistency_result = check_with_gpt4o(client, first_caption, samples, args.model)
        result = {
            "id": image_id,
            "ground_truth": ground_truth,
            "weather": data.get("weather", "unknown"),
            "dark": data.get("dark", False),
            "first_caption": first_caption,
            "samples": samples,
            "consistency_result": consistency_result
        }
        
        results.append(result)
        processed_images[image_id] = True
        last_processed_index = i
        save_progress_state(results, last_processed_index, None)
        time.sleep(1)
    
    print(f"All hallucination checks completed and saved to {SELFCHECK_RESULTS_PATH}")
    print(f"Total tokens used: {total_tokens_used}")

if __name__ == "__main__":
    main()