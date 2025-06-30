import os
import pandas as pd
import json
import time
import concurrent.futures
from dotenv import load_dotenv
import requests
import replicate
from openai import OpenAI
from PIL import Image
from io import BytesIO
import itertools
import random

# Load environment variables
load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
IDEOGRAM_API_KEY = os.getenv("IDEOGRAM_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Set environment variables for clients
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# Set up clients
openai_client = OpenAI()

# Configuration
CSV_FILE_PATH = "./data UAI/MutliModalSocialBias/Grammar is Bias ?/Dataset/ReadyToUse/GenderMismatchWithPromptsWithRussianFull200V4.csv"
BASE_DIR = "./data UAI/MutliModalSocialBias/Grammar is Bias ?/PromptToImage/ModelsOutputForPaperEvalution/"
MAX_RETRIES = 3
RETRY_DELAY = 90
NUM_SHOTS = 4  # Number of images per prompt
BATCH_SIZE = 12  # Number of requests to make per model batch

# API Rate limits (requests per minute)
API_RATE_LIMITS = {
    "dalle3": 15,    # Adjust based on your OpenAI rate limits
    "ideogram": 20,  # Adjust based on your Ideogram rate limits
    "flux": 20       # Adjust based on your Replicate rate limits
}

# Create base directory
os.makedirs(BASE_DIR, exist_ok=True)

def clean_folder_name(text):
    if not text:
        return "unknown"
    cleaned = ''.join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in str(text))
    cleaned = cleaned.replace(' ', '_').lower()
    return cleaned[:50] if len(cleaned) > 50 else cleaned

def is_valid_url(url):
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return 200 <= response.status_code < 400
    except Exception:
        return False

def save_image_from_url(url, filepath):
    try:
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(filepath)
            print(f"✅ Saved: {filepath}")
            return True
    except Exception as e:
        print(f"❌ Error saving image from {url}: {e}")
        return False

def generate_dalle_image(prompt, filepath):
    if os.path.exists(filepath):
        return filepath
    for attempt in range(MAX_RETRIES):
        try:
            response = openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            image_url = response.data[0].url
            if save_image_from_url(image_url, filepath):
                return filepath
        except Exception as e:
            print(f"⚠️ DALL-E 3 error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)
    return None

def generate_ideogram_image(prompt, filepath):
    if os.path.exists(filepath):
        return filepath
    url = "https://api.ideogram.ai/v1/ideogram-v3/generate"
    headers = {"Api-Key": IDEOGRAM_API_KEY, "Content-Type": "application/json"}
    data = {"prompt": prompt, "rendering_speed": "DEFAULT", "aspect_ratio": "1x1"}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                image_url = result["data"][0]["url"]
                if save_image_from_url(image_url, filepath):
                    return filepath
            else:
                print(f"⚠️ Ideogram error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"⚠️ Ideogram error: {e}")
        time.sleep(RETRY_DELAY)
    return None

def generate_flux_image(prompt, filepath):
    if os.path.exists(filepath):
        return filepath
    for attempt in range(MAX_RETRIES):
        try:
            output = replicate.run(
                "black-forest-labs/flux-1.1-pro",
                input={
                    "prompt": prompt,
                    "aspect_ratio": "1:1",
                    "output_format": "png",
                    "output_quality": 90,
                    "safety_tolerance": 6,
                    "prompt_upsampling": True
                }
            )
            image_url = str(output) if not isinstance(output, list) else output[0]
            if is_valid_url(image_url) and save_image_from_url(image_url, filepath):
                return filepath
        except Exception as e:
            print(f"⚠️ Flux error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
        time.sleep(RETRY_DELAY)
    return None

def process_prompt(args):
    model_name, prompt, filepath = args
    generator_func = model_generators.get(model_name)
    return generator_func(prompt, filepath) if generator_func else None

model_generators = {
    "dalle3": generate_dalle_image,
    "ideogram": generate_ideogram_image,
    "flux": generate_flux_image
}

def generate_shot_folders(language, model_name, gender, bias_type, word, language_types, all_tasks):
    # Create the base path following the requested structure
    # language/model/gender/bias_type/word/
    base_path = os.path.join(BASE_DIR, language, model_name, gender, bias_type, word)
    os.makedirs(base_path, exist_ok=True)
    
    # Save general information
    info_data = {
        "word": word,
        "language": language,
        "model": model_name,
        "gender": gender,
        "bias_type": bias_type,
        "language_types": language_types
    }
    with open(os.path.join(base_path, "info.json"), 'w', encoding='utf-8') as f:
        json.dump(info_data, f, ensure_ascii=False, indent=2)
    
    for lang_type, prompts in language_types.items():
        if not prompts:
            continue
        
        # Create language type directory (english, native, chinese)
        lang_dir = os.path.join(base_path, lang_type)
        os.makedirs(lang_dir, exist_ok=True)
        
        # Save prompts.json in the language directory
        with open(os.path.join(lang_dir, "prompts.json"), 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        
        # Create shot directories
        for shot_idx in range(NUM_SHOTS):
            shot_dir = os.path.join(lang_dir, f"shot_{shot_idx + 1}")
            os.makedirs(shot_dir, exist_ok=True)
            
            # Queue up image generation tasks
            for prompt_idx, prompt in enumerate(prompts):
                image_path = os.path.join(shot_dir, f"{model_name}_prompt_{prompt_idx + 1}.png")
                if not os.path.exists(image_path):
                    all_tasks.append((model_name, prompt, image_path))
    
    return all_tasks

def batch_tasks_by_model(tasks):
    # Group tasks by model
    model_tasks = {}
    for task in tasks:
        model_name = task[0]
        if model_name not in model_tasks:
            model_tasks[model_name] = []
        model_tasks[model_name].append(task)
    
    # Shuffle tasks within each model to diversify the content
    for model_name in model_tasks:
        random.shuffle(model_tasks[model_name])
    
    # Create batches for each model
    all_batches = []
    for model_name, model_tasks_list in model_tasks.items():
        for i in range(0, len(model_tasks_list), BATCH_SIZE):
            batch = model_tasks_list[i:i+BATCH_SIZE]
            all_batches.append((model_name, batch))
    
    # Shuffle batches to alternate between models
    random.shuffle(all_batches)
    return all_batches

def process_batch(batch_data):
    model_name, tasks = batch_data
    results = []
    for task in tasks:
        try:
            result = process_prompt(task)
            results.append(result)
            # Add a small delay between requests to the same API to avoid rate limits
            time.sleep(60 / API_RATE_LIMITS.get(model_name, 20))
        except Exception as e:
            print(f"Error processing task {task}: {e}")
    return results

def main():
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Loaded {len(df)} rows from CSV")
        all_tasks = []
        
        for idx, row in df.iterrows():
            # Fix: Convert idx+1 and len(df) to strings before concatenation
            print("Processing row {}/{}".format(idx+1, len(df)))
            
            # Check if row is valid
            is_valid = str(row.get('is valid', '')).lower()
            if is_valid != 'true':
                print(f"Skipping row {str(idx+1)} - Not valid")
                continue
                
            language = clean_folder_name(row.get('Language Source', ''))
            gender = clean_folder_name(row.get('Grammatical Gender', ''))
            bias_type = clean_folder_name(row.get('Majority Bias Type', ''))
            word = clean_folder_name(row.get('Native Language Word', ''))
            
            # Process prompts safely
            def safe_eval_prompts(value):
                if pd.isna(value):
                    return []
                
                if isinstance(value, str):
                    # Try to evaluate as string representation of list
                    try:
                        result = eval(value)
                        if isinstance(result, list):
                            return result
                        else:
                            return [str(result)]
                    except:
                        # If eval fails, check if it's JSON-like
                        if value.startswith('[') and value.endswith(']'):
                            try:
                                import json
                                result = json.loads(value.replace("'", '"'))
                                return result
                            except:
                                pass
                        # Return as single item list if all else fails
                        return [value]
                else:
                    # Not a string, might be already a list
                    if isinstance(value, list):
                        return value
                    else:
                        return [str(value)]
            
            # Apply the safe evaluation function
            native_prompts = safe_eval_prompts(row.get('Native Templates', '[]'))
            english_prompts = safe_eval_prompts(row.get('English Templates', '[]'))
            chinese_prompts = safe_eval_prompts(row.get('Chinese Templates', '[]'))
            
            # Clean prompts
            native_prompts = [p.strip('"\'') for p in native_prompts if p and str(p).strip()]
            english_prompts = [p.strip('"\'') for p in english_prompts if p and str(p).strip()]
            chinese_prompts = [p.strip('"\'') for p in chinese_prompts if p and str(p).strip()]
            
            language_types = {
                "english": english_prompts,
                "native": native_prompts,
                "chinese": chinese_prompts
            }
            
            # Process for each model
            for model_name in model_generators.keys():
                all_tasks = generate_shot_folders(
                    language, model_name, gender, bias_type, word, 
                    language_types, all_tasks
                )
        
        
        if not all_tasks:
            print("No images to generate. All shots exist.")
            return
        
        # Group tasks by model and create batches
        batches = batch_tasks_by_model(all_tasks)
        
        print(f"Starting {str(len(all_tasks))} image generation tasks in {str(len(batches))} batches...")
        
        # Process batches with controlled concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    future.result()
                    print(f"Progress: {str(i)}/{str(len(batches))} batches completed")
                except Exception as e:
                    print(f"Batch failed: {e}")
        
        print("✅ All image generation tasks completed.")
        print(f"Images saved in: {os.path.abspath(BASE_DIR)}")
    
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt received. Exiting gracefully...")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()






