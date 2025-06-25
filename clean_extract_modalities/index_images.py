import os
import json
import sys

def index_images(root_folder, output_file):
    image_paths = {}
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_paths[filename.lower()] = os.path.join(dirpath, filename)
    print(f"Indexed {len(image_paths)} images.")
    with open(output_file, 'w') as f:
        json.dump(image_paths, f)

if __name__== "_main_":
    IMAGE_ROOT = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]
    index_images(IMAGE_ROOT, OUTPUT_FILE)