import os
import json
import time
import base64
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

LABELS_PATH = "labels_processed.json"
IMAGES_FOLDER = "saved"

RELEVANT_OBJECTS = [
    "car", "bus", "truck", "bicycle", "motorcycle", 
    "pedestrian", "traffic light red", "traffic light green", 
    "traffic light yellow", "speed bump"
]

TOKEN_LIMIT = 800000  # 500k token limit per session
total_tokens_used = 0

def encode_image(image_path):
    """Encode image as base64 for API request"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_prompt():
    """Create a clear and structured prompt as described in the paper"""
    prompt = f"""
    You are analyzing a traffic scene image for automotive perception.
    
    Predefined relevant objects: {', '.join(RELEVANT_OBJECTS)}
    
    IMPORTANT RULES THAT YOU MUST FOLLOW:
    1. Only detect objects that are at the same level as the vehicle taking the photo (objects on the road).
    2. Do NOT include objects or pedestrians on sidewalks.
    3. The ONLY exception is traffic lights, which should always be detected regardless of position as long as you can distinguish their color.
    4. Return a list of each object from the list of predefined relevant objects that you see in this image.
    5. If you identify multiple objects of the same type, you must output their occurrence an equal number of times to the observed cardinality.
    6. In the case that you do not see any relevant objects, simply return an empty output.
    7. Your response must be a comma-separated list of detected objects, with no additional text, formatting, or code blocks. Example: car, car, traffic light red
    8. Be precise and thorough in your detection. Do not include objects that aren't clearly visible.
    """
    return prompt.strip()

def query_model(image_path, model_name):
    """Query the specified OpenAI model with the image and return the raw output"""
    global total_tokens_used
    
    try:
        base64_image = encode_image(image_path)
        prompt = create_prompt()
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a perception system for autonomous vehicles."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=300
        )
        tokens_in_request = response.usage.total_tokens
        total_tokens_used += tokens_in_request
        print(f"  Tokens used: {tokens_in_request} | Total used: {total_tokens_used} | Remaining: {TOKEN_LIMIT - total_tokens_used}")
        return response.choices[0].message.content.strip()
            
    except Exception as e:
        print(f"Error querying model {model_name}: {str(e)}")
        return f"ERROR: {str(e)}"

def save_progress_state(output_path, outputs, last_processed_index):
    """Save the current progress state to a file"""
    progress_path = f"{output_path}.progress"
    progress_state = {
        "last_processed_index": last_processed_index,
        "total_tokens_used": total_tokens_used
    }
    
    with open(progress_path, 'w') as f:
        json.dump(progress_state, f)
    with open(output_path, 'w') as f:
        json.dump(outputs, f, indent=2)

def load_progress_state(output_path):
    """Load the previous progress state if it exists"""
    global total_tokens_used
    
    progress_path = f"{output_path}.progress"
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress_state = json.load(f)
            total_tokens_used = progress_state.get("total_tokens_used", 0)
            return progress_state.get("last_processed_index", -1)
    return -1

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 prompter.py <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    OUTPUT_PATH = f"model_outputs_{model_name}.json"
    outputs = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, 'r') as f:
            outputs = json.load(f)
    processed_images = {item["id"]: item for item in outputs}
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
    last_processed_index = load_progress_state(OUTPUT_PATH)
    
    print(f"Starting with {total_tokens_used} tokens already used. Limit: {TOKEN_LIMIT}")
    total = len(labels)
    for i, label in enumerate(labels):
        if i <= last_processed_index:
            print(f"Skipping index {i} (already processed in previous run)")
            continue
        
        image_id = label["id"]
        image_path = os.path.join(IMAGES_FOLDER, image_id)
        if image_id in processed_images:
            print(f"Skipping {image_id} - already in output file")
            continue
        
        print(f"Processing {i+1}/{total}: {image_id}")
        if total_tokens_used >= TOKEN_LIMIT:
            print(f"Token limit of {TOKEN_LIMIT} reached. Saving progress and exiting.")
            save_progress_state(OUTPUT_PATH, outputs, i-1)
            sys.exit(0)
        max_retries = 3
        for retry in range(max_retries):
            try:
                raw_output = query_model(image_path, model_name)
                new_item = {
                    "id": image_id,
                    "raw_output": raw_output
                }
                
                outputs.append(new_item)
                processed_images[image_id] = new_item
                save_progress_state(OUTPUT_PATH, outputs, i)
                time.sleep(2)
                break  # Successful, exit retry loop
                
            except Exception as e:
                print(f"    Error (attempt {retry+1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    wait_time = (retry + 1) * 5  # Exponential backoff
                    print(f"    Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"    Failed to process after {max_retries} attempts")
                    save_progress_state(OUTPUT_PATH, outputs, i-1)

    print(f"Processing complete. Results saved to {OUTPUT_PATH}")
    print(f"Total tokens used in this session: {total_tokens_used}")

if __name__ == "__main__":
    main()