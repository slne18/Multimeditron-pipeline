import json
import base64
from PIL import Image
from io import BytesIO
import os

# Path to the input JSONL file containing the API requests with base64-encoded images
jsonl_path = "batches_eval/part_1.jsonl"

# Helper function to check if a base64 string represents a valid image
def is_valid_base64_image(image_b64):
    try:
        decoded = base64.b64decode(image_b64)
        Image.open(BytesIO(decoded))  # Try to open the decoded image
        return True
    except Exception as e:
        return False

# First pass: check validity of all images in the JSONL file
with open(jsonl_path, "r") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        try:
            image_url = data["body"]["messages"][1]["content"][1]["image_url"]["url"]
            if image_url.startswith("data:image/jpeg;base64,"):
                image_b64 = image_url.split(",")[1]
                valid = is_valid_base64_image(image_b64)
                print(f"Line {i + 1}: {'OK' if valid else 'Invalid image'}")
            else:
                print(f"Line {i + 1}: Incorrect format (not base64)")
        except Exception as e:
            print(f"Line {i + 1}: Error during extraction: {e}")

# Directory where valid images will be saved
output_dir = "preview_images"
os.makedirs(output_dir, exist_ok=True)

saved_count = 0

# Second pass: save the first 5 valid images
with open(jsonl_path, "r") as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            image_url = data["body"]["messages"][1]["content"][1]["image_url"]["url"]
            if image_url.startswith("data:image/jpeg;base64,"):
                image_b64 = image_url.split(",")[1]
                image_data = base64.b64decode(image_b64)
                img = Image.open(BytesIO(image_data))
                img.save(os.path.join(output_dir, f"image_{saved_count + 1}.jpg"))
                print(f"Image {saved_count + 1} saved successfully (line {i + 1}).")
                saved_count += 1
                if saved_count >= 5:
                    break
        except Exception as e:
            print(f"Error on line {i + 1}: {e}")
