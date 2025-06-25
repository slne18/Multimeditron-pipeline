import argparse
import jsonlines
import os

def extract_data(entry):
    """Extract image name and caption, converting conversations to text format."""
    image_name = entry.get("modalities", [{}])[0].get("value", "unknown")
    
    caption = ""  

    # Convert "conversations" format to "text"
    if "conversations" in entry:
        # Extract assistant's response
        for convo in entry["conversations"]:
            if convo.get("role") == "assistant":
                caption = convo.get("content", "")
                break  # Stop at the first assistant response

        # Replace "conversations" with "text"
        entry = {
            "text": caption,
            "modalities": entry.get("modalities", [])
        }

    # Keep "text" format as is, just clean up the special token if necessary
    elif "text" in entry:
        caption = entry["text"]
        if caption.endswith("<|reserved_special_token_0|>"):
            caption = caption[:-len("<|reserved_special_token_0|>")]
        entry["text"] = caption

    else:
        entry = {} 

    return image_name, caption, entry


def load_and_process_data(files):
    """Load data from JSONL files, remove duplicates with identical captions, and store cleaned data."""
    image_entries = {}  # Stores image name -> (caption, file, entry)
    file_data = {file_path: [] for file_path in files}  
    duplicates = []  # Track duplicates found

    for file_path in files:
        with jsonlines.open(file_path, 'r') as reader:
            for obj in reader:
                image_name, caption, transformed_entry = extract_data(obj)

                if image_name in image_entries:
                    existing_caption, existing_file, _ = image_entries[image_name]

                    # Check if captions are identical
                    if caption == existing_caption:
                        duplicates.append(image_name) 
                        print(f"🛑 Duplicate found: {image_name}")
                        continue  # Skip this duplicate entry

                # Store unique entry
                image_entries[image_name] = (caption, file_path, transformed_entry)
                file_data[file_path].append(transformed_entry)

    return file_data, duplicates

def save_deduplicated_data(file_data, original_file_paths, duplicates):
    """Save cleaned data into new JSONL files, always saving the cleaned dataset."""
    for original_file, entries in file_data.items():
        # Create 'clean_dataset' folder in the same location as the input file
        directory = os.path.dirname(original_file)
        clean_dataset_dir = os.path.join(directory, 'clean_dataset')

        # Create 'clean_dataset' folder if it doesn't exist
        if not os.path.exists(clean_dataset_dir):
            os.makedirs(clean_dataset_dir)

        new_file = os.path.join(clean_dataset_dir, os.path.basename(original_file))

        # Save the dataset, regardless of whether duplicates were found or not
        with jsonlines.open(new_file, 'w') as writer:
            writer.write_all(entries)
        print(f"✅ Cleaned file saved as: {new_file}")

    # Print if no duplicates were found
    if not duplicates:
        print("✅ No duplicates found.")

def get_jsonl_files(paths):
    """Get all .jsonl files from the given list of paths (files or directories)."""
    files = []
    for path in paths:
        if os.path.isdir(path):
            # If it's a directory, get all .jsonl files recursively
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith(".jsonl"):
                        files.append(os.path.join(root, filename))
        elif os.path.isfile(path) and path.endswith(".jsonl"):
            # If it's a file and ends with .jsonl, add it
            files.append(path)
        else:
            print(f"⚠️ Skipping non-JSONL file or invalid directory: {path}")
    return files

def main():
    parser = argparse.ArgumentParser(description="Remove duplicate images with identical captions from JSONL files.")
    parser.add_argument('paths', nargs='+', help='List of JSONL files or directories to process')
    args = parser.parse_args()

    # Get all JSONL files from the provided paths (files or directories)
    files = get_jsonl_files(args.paths)

    if files:
        file_data, duplicates = load_and_process_data(files)
        save_deduplicated_data(file_data, files, duplicates)  
    else:
        print("No valid JSONL files found to process.")

if __name__ == "__main__":
    main()
