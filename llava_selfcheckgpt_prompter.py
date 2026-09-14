import os
import json
import time
import base64
import requests
import subprocess
import argparse

LABELS_PATH = "labels_processed.json"
IMAGES_FOLDER = "saved"
CAPTIONS_OUTPUT_PATH = "llava_captions.json"

RELEVANT_OBJECTS = [
    "car", "bus", "truck", "bicycle", "motorcycle", 
    "pedestrian", "traffic light red", "traffic light green", 
    "traffic light yellow", "speed bump"
]

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

def query_llava(image_path, model_name="llava"):
    """Query the LLaVA model to generate captions for an image"""
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
    parser = argparse.ArgumentParser(description="Generate captions using LLaVA")
    parser.add_argument("--model", default="llava", help="Model name to use")
    parser.add_argument("--samples", type=int, default=5, help="Number of caption samples to generate per image")
    parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
    args = parser.parse_args()
    pull_model(args.model)
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)
    if args.max_images is not None:
        labels = labels[:args.max_images]
    results = []
    total_images = len(labels)
    for i, label in enumerate(labels):
        image_id = label["id"]
        image_path = os.path.join(IMAGES_FOLDER, image_id)
        
        print(f"Processing {i+1}/{total_images}: {image_id}")
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        caption_samples = []
        first_caption = None
        
        for j in range(args.samples):
            print(f"  Generating sample {j+1}/{args.samples}")
            caption = query_llava(image_path, args.model)
            if j == 0:
                first_caption = caption
            else:
                caption_samples.append(caption)
            time.sleep(1)
        result = {
            "id": image_id,
            "ground_truth": label["objects"],
            "weather": label.get("weather", "unknown"),
            "dark": label.get("dark", False),
            "first_caption": first_caption,
            "samples": caption_samples  # All samples except the first one
        }
        results.append(result)
        if (i + 1) % 10 == 0 or (i + 1) == total_images:
            with open(CAPTIONS_OUTPUT_PATH, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Progress saved: {i+1}/{total_images} images processed")
    
    print(f"All captions generated and saved to {CAPTIONS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()