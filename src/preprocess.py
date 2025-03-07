import os
import json
import argparse
from datasets import Dataset
import ipdb

def parse_arguments():
    """
    Parse command line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Process dataset by truncating content in chosen and rejected fields.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the original dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the modified dataset")
    parser.add_argument("--truncate_ratio", type=float, default=0.50, help="Ratio for truncating content (default: 0.40)")
    return parser.parse_args()

def modify_sample(sample, truncate_ratio):
    """
    Modify each data point by truncating the 'content' field in the last dictionary of 'rejected' and 'chosen' lists.
    """
    def truncate_last_content(dialog_list):
        """
        Truncate the 'content' field of the last dictionary in the list to the specified ratio.
        """
        if dialog_list and isinstance(dialog_list[-1], dict) and "content" in dialog_list[-1]:
            content = dialog_list[-1]["content"]
            mid = int(len(content) * truncate_ratio)  # Truncate content based on the specified ratio
            dialog_list[-1]["content"] = content[:mid]
        return dialog_list

    # Modify 'rejected' and 'chosen' lists
    sample["rejected"] = truncate_last_content(sample.get("rejected", []))
    sample["chosen"] = truncate_last_content(sample.get("chosen", []))

    return sample

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create new dataset directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load original dataset_info.json
    dataset_info_path = os.path.join(args.input_dir, "dataset_info.json")
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    
    # Process dataset files
    new_file_info = {}  # For updating dataset_info.json
    for file_name, file_info in dataset_info["download_checksums"].items():
        try:
            original_file_path = file_name
            new_file_path = os.path.join(args.output_dir, os.path.basename(file_name))
            print(f"Processing file: {original_file_path}")
            
            # Load original dataset
            dataset = Dataset.from_file(original_file_path)
            
            # Apply modification function with the specified truncation ratio
            modified_dataset = dataset.map(
                lambda sample: modify_sample(sample, args.truncate_ratio), 
                num_proc=8
            )
            
            # Save modified dataset
            modified_dataset.save_to_disk(new_file_path)
            print(f"Saved modified dataset to: {new_file_path}")
            
            # Update file information
            new_file_info[new_file_path] = {
                "num_bytes": os.path.getsize(new_file_path),
                "checksum": None  # Calculate checksum if needed
            }
        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")
    
    # Update dataset_info.json
    dataset_info["download_checksums"] = new_file_info
    dataset_info["dataset_size"] = sum(info["num_bytes"] for info in new_file_info.values())
    dataset_info["size_in_bytes"] = dataset_info["dataset_size"] + dataset_info.get("download_size", 0)
    
    # Save updated dataset_info.json
    new_dataset_info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(new_dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=4)
    
    print(f"New dataset organized and saved at: {args.output_dir}")
    print(f"Applied truncation ratio: {args.truncate_ratio}")

if __name__ == "__main__":
    main()