import os
import json
import time
import base64
import requests
import sys
import subprocess

LABELS_PATH = "labels_processed.json"
IMAGES_FOLDER = "saved"

RELEVANT_OBJECTS = [
    "car", "bus", "truck", "bicycle", "motorcycle", 
    "pedestrian", "traffic light red", "traffic light green", 
    "traffic light yellow", "speed bump"
]
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

def check_model_exists(model_name):
    """Check if the model exists in Ollama"""
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    return model_name in result.stdout

def pull_model(model_name):
    """Pull the model if it doesn't exist"""
    if not check_model_exists(model_name):
        print(f"Pulling {model_name} model...")
        subprocess.run(['ollama', 'pull', model_name], check=True)
    else:
        print(f"{model_name} model already exists")
    return model_name

def query_ollama(image_path, model_name):
    """Query the Ollama model with the image"""
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        prompt = create_prompt()
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model_name,
                'prompt': prompt,
                'images': [base64_image],
                'stream': False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['response'].strip()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return f"ERROR: API call failed with status {response.status_code}"
            
    except Exception as e:
        print(f"Error querying model {model_name}: {str(e)}")
        return f"ERROR: {str(e)}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 llava_prompter.py <model_name>")
        print("Example: python3 llava_prompter.py llava")
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
    pull_model(model_name)
    total = len(labels)
    for i, label in enumerate(labels):
        image_id = label["id"]
        image_path = os.path.join(IMAGES_FOLDER, image_id)
        if image_id in processed_images:
            print(f"Skipping {image_id} - already processed")
            continue
        
        print(f"Processing {i+1}/{total}: {image_id}")
        max_retries = 3
        for retry in range(max_retries):
            try:
                raw_output = query_ollama(image_path, model_name)
                new_item = {
                    "id": image_id,
                    "raw_output": raw_output
                }
                
                outputs.append(new_item)
                processed_images[image_id] = new_item
                with open(OUTPUT_PATH, 'w') as f:
                    json.dump(outputs, f, indent=2)
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

    print(f"Processing complete. Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()