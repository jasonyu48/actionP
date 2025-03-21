import os
import numpy as np
from tqdm import tqdm
import shutil

# process the data in the xrf555 folder so that the second dimension is 30

def process_xrf_data():
    # Base directory containing all action types
    base_dir = 'xrf555'
    # Output directory for processed data
    output_base_dir = 'xrf555_processed'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # Get all action types (folders in the base directory)
    action_types = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Statistics counters
    total_files = 0
    padded_files = 0
    truncated_files = 0
    unchanged_files = 0
    
    # Process each action type
    for action_type in action_types:
        print(f"Processing action type: {action_type}")
        action_path = os.path.join(base_dir, action_type)
        output_action_path = os.path.join(output_base_dir, action_type)
        
        # Create output action directory if it doesn't exist
        if not os.path.exists(output_action_path):
            os.makedirs(output_action_path)
        
        # Navigate to the 90 folder
        ninety_path = os.path.join(action_path, '90')
        output_ninety_path = os.path.join(output_action_path, '90')
        
        if not os.path.exists(ninety_path):
            print(f"Warning: No '90' folder found for action type {action_type}")
            continue
        
        # Create output 90 directory if it doesn't exist
        if not os.path.exists(output_ninety_path):
            os.makedirs(output_ninety_path)
        
        # Get all numbered folders
        numbered_folders = [d for d in os.listdir(ninety_path) if os.path.isdir(os.path.join(ninety_path, d))]
        
        # Process each numbered folder
        for folder in tqdm(numbered_folders, desc=f"Folders in {action_type}"):
            folder_path = os.path.join(ninety_path, folder)
            output_folder_path = os.path.join(output_ninety_path, folder)
            
            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            
            # Process xrf_specs.npy
            xrf_specs_path = os.path.join(folder_path, 'xrf_specs.npy')
            output_xrf_specs_path = os.path.join(output_folder_path, 'xrf_specs.npy')
            
            # Process xrf_sim2real_specs.npy
            xrf_sim2real_specs_path = os.path.join(folder_path, 'xrf_sim2real_specs.npy')
            output_xrf_sim2real_specs_path = os.path.join(output_folder_path, 'xrf_sim2real_specs.npy')
            
            # Copy other files directly
            for file_name in os.listdir(folder_path):
                if file_name not in ['xrf_specs.npy', 'xrf_sim2real_specs.npy']:
                    src_file = os.path.join(folder_path, file_name)
                    dst_file = os.path.join(output_folder_path, file_name)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
            
            # Process both files
            for src_path, dst_path in [(xrf_specs_path, output_xrf_specs_path), 
                                      (xrf_sim2real_specs_path, output_xrf_sim2real_specs_path)]:
                if os.path.exists(src_path):
                    try:
                        # Load the data
                        data = np.load(src_path)
                        total_files += 1
                        
                        # Get the current second dimension size
                        second_dim_size = data.shape[1]
                        
                        # Process based on second dimension size
                        if second_dim_size < 30:
                            # Repeat along second dimension to make it 30
                            repeats_needed = int(np.ceil(30 / second_dim_size))
                            repeated_data = np.repeat(data, repeats_needed, axis=1)
                            # Truncate to exactly 30 in case of over-repetition
                            processed_data = repeated_data[:, :30, :, :]
                            padded_files += 1
                        elif second_dim_size > 30:
                            # Truncate to keep only first 30 elements
                            processed_data = data[:, :30, :, :]
                            truncated_files += 1
                        else:
                            # Already 30, no change needed
                            processed_data = data
                            unchanged_files += 1
                        
                        # Save the processed data
                        np.save(dst_path, processed_data)
                        
                    except Exception as e:
                        print(f"Error processing {src_path}: {e}")
    
    # Print statistics
    print("\nProcessing Statistics:")
    print(f"Total files processed: {total_files}")
    print(f"Files padded (second dim < 30): {padded_files}")
    print(f"Files truncated (second dim > 30): {truncated_files}")
    print(f"Files unchanged (second dim = 30): {unchanged_files}")
    print(f"\nProcessed data saved to: {output_base_dir}")

if __name__ == "__main__":
    process_xrf_data() 