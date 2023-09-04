import requests
import os
import json
from PIL import Image
from io import BytesIO
import random

# Initialize parameters
api_url = "https://datasets-server.huggingface.co/rows"
params = {
    "dataset": "kakaobrain/coyo-700m",
    "config": "kakaobrain--coyo-700m",
    "split": "train",
    "offset": None
}

# Directory to save images and text
base_dir = "train_dataset_01"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "image"), exist_ok=True)

# Load existing filter_cap.json or initialize an empty one
json_file_path = os.path.join(base_dir, "filter_cap.json")
if os.path.exists(json_file_path):
    with open(json_file_path, "r") as f:
        filter_cap = json.load(f)
else:
    filter_cap = {"annotations": []}

# Initialize random offset
all_offsets = random.sample(range(1, 7000000), 100)  # Assuming there are at least 10,000 records, adjust as needed

# Fetch 5000 images by making 50 API calls with random offsets
count = 0
for offset in all_offsets:
    params["offset"] = offset
    response = requests.get(api_url, params=params)
    data = response.json()
    rows = data.get("rows", [])
    for row in rows:
        row_data = row.get("row", {})
        img_url = row_data.get("url", "")
        text = row_data.get("text", "")
        image_name = f"{count}.jpg"

        try:
            # Download and save the image
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            img.save(os.path.join(base_dir, "image", image_name))

            # Append the text data to filter_cap.json
            filter_cap["annotations"].append({
                "image_id": str(count),
                "caption": text
            })

            print(f"Saved {count}.jpg and corresponding text")
            count += 1

            if count >= 500:
                break
        except Exception as e:
            print(f"Could not save {image_name} due to {e}")

    if count >= 500:
        break

# Save the updated filter_cap.json
with open(json_file_path, "w") as f:
    json.dump(filter_cap, f)
