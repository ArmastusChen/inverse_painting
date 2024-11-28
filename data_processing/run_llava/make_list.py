import json
import os
import cv2
import numpy as np
import tqdm
from pathlib import Path
import argparse

# Function to retrieve all .json files recursively
def get_all_files(src_dir, extension="*.json"):
    src_dir = Path(src_dir)
    all_files = list(src_dir.rglob(extension))
    return [str(file) for file in all_files]  # Convert to string

# Argument parser
parser = argparse.ArgumentParser(description="Process JSON files to create concatenated images and generate a conversation dataset.")
parser.add_argument("--split", type=str, default="train", help="Dataset split to process (e.g., train, val, test).")
parser.add_argument("--root_dir", type=str, default="../../data/sample_data", help="Root directory containing the dataset.")
parser.add_argument("--prompt", type=str, default="There are two images side by side. The left image is an intermediate stage in a painting process of the right image. Please tell me what content should be painted next? The answer should be less than 2 words.", 
                    help="Prompt to include in the conversations.")
args = parser.parse_args()

# Use parsed arguments
split = args.split
root_dir = args.root_dir
prompt = args.prompt

# Get all JSON files
json_files = get_all_files(f'{root_dir}/{split}/text')
json_files = [json_file.replace('/white_', '/0_') for json_file in json_files]
json_files = sorted(json_files, key=lambda x: (x.split('/')[-2], int(x.split('/')[-1].split('_')[0])))
json_files = [json_file.replace('/0_', '/white_') for json_file in json_files]

# Initialize storage for the new JSON data
json_info = []
cnt = 0

# Process each JSON file
for json_file in tqdm.tqdm(json_files):
    with open(json_file, 'r') as f:
        data = json.load(f)

    json_name = json_file.split('/')[-1]
    cur_img_name = data['cur_image_name']
    ref_img_name = data['ref_img_name']
    next_text_corrected = data['next_text']

    # Image paths
    cur_img_path = json_file.replace('/text/', '/rgb/').replace(json_name, cur_img_name + '.jpg')
    ref_img_path = json_file.replace('/text/', '/rgb/').replace(json_name, ref_img_name + '.jpg')

    # Load images
    ref_img = cv2.imread(ref_img_path)
    if 'white' in cur_img_name:
        cur_img = 255 * np.ones_like(ref_img, dtype=np.uint8)
    else:
        cur_img = cv2.imread(cur_img_path)

    # Concatenate images horizontally
    query_image = cv2.hconcat([cur_img, ref_img])

    # Save concatenated image
    
    dst_dir = os.path.dirname(json_file).replace('/text/', '/llava_image/')
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f'{cur_img_name}_{ref_img_name}.png')
    cv2.imwrite(dst_path, query_image)

    json_image_path = dst_path.split('llava_image/')[-1]

    # Create conversation entry
    json_cur_dict = {
        'id': cnt,
        "image": json_image_path,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + prompt
            },
            {
                "from": "gpt",
                "value": next_text_corrected
            },
        ]
    }

    cnt += 1
    json_info.append(json_cur_dict)

# Write all conversation data to a single JSON file
dst_json_path = f'{root_dir}/{split}/llava_json.json'
os.makedirs(os.path.dirname(dst_json_path), exist_ok=True)
with open(dst_json_path, 'w') as f:
    json.dump(json_info, f, indent=4)

print(f"Generated JSON saved to {dst_json_path}")
