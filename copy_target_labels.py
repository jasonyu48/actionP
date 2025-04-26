#!/usr/bin/env python3
import os
import shutil
import argparse
import glob
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Copy target_labels.npy files from original dataset to processed dataset')
    parser.add_argument('--src_dir', type=str, default='/weka/scratch/rzhao36/lwang/datasets/HOI/RealAction/datasets/focus',
                        help='Source directory containing original dataset')
    parser.add_argument('--dst_dir', type=str, default='/scratch/tshu2/jyu197/Focus_processed_multi_angle_rfid_real',
                        help='Destination directory containing processed dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    
    # Verify source directory exists
    if not os.path.exists(src_dir):
        print(f"Error: Source directory {src_dir} does not exist")
        return
    
    # Verify destination directory exists
    if not os.path.exists(dst_dir):
        print(f"Error: Destination directory {dst_dir} does not exist")
        return
    
    # Get all action types (directories in the source directory)
    action_types = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    print(f"Found {len(action_types)} action types in source directory")
    
    # Track statistics
    files_copied = 0
    files_already_exist = 0
    
    # Process each action type
    for action_type in tqdm(action_types, desc="Processing action types"):
        # Check if action folder exists in destination
        dst_action_dir = os.path.join(dst_dir, action_type)
        if not os.path.exists(dst_action_dir):
            print(f"Warning: Action directory {dst_action_dir} does not exist, creating it")
            os.makedirs(dst_action_dir, exist_ok=True)
        
        # Get angle folders (0, 90, 180, 270)
        src_action_dir = os.path.join(src_dir, action_type)
        angle_folders = [d for d in os.listdir(src_action_dir) 
                        if os.path.isdir(os.path.join(src_action_dir, d)) 
                        and d in ['0', '90', '180', '270']]
        
        # Process each angle folder
        for angle in angle_folders:
            src_angle_dir = os.path.join(src_action_dir, angle)
            dst_angle_dir = os.path.join(dst_action_dir, angle)
            
            # Check if angle folder exists in destination
            if not os.path.exists(dst_angle_dir):
                print(f"Warning: Angle directory {dst_angle_dir} does not exist, creating it")
                os.makedirs(dst_angle_dir, exist_ok=True)
            
            # Copy target_labels.npy if it exists in source
            src_labels_file = os.path.join(src_angle_dir, 'target_labels.npy')
            dst_labels_file = os.path.join(dst_angle_dir, 'target_labels.npy')
            
            if os.path.exists(src_labels_file):
                if os.path.exists(dst_labels_file):
                    files_already_exist += 1
                else:
                    shutil.copy2(src_labels_file, dst_labels_file)
                    files_copied += 1
    
    print(f"Copy operation completed.")
    print(f"Files copied: {files_copied}")
    print(f"Files already existed: {files_already_exist}")
    
    # Verify copied files
    print("\nVerifying destination directories have target_labels.npy files...")
    
    dst_action_types = [d for d in os.listdir(dst_dir) if os.path.isdir(os.path.join(dst_dir, d))]
    missing_files = 0
    
    for action_type in dst_action_types:
        dst_action_dir = os.path.join(dst_dir, action_type)
        angle_folders = [d for d in os.listdir(dst_action_dir) 
                        if os.path.isdir(os.path.join(dst_action_dir, d)) 
                        and d in ['0', '90', '180', '270']]
        
        for angle in angle_folders:
            dst_angle_dir = os.path.join(dst_action_dir, angle)
            dst_labels_file = os.path.join(dst_angle_dir, 'target_labels.npy')
            
            if not os.path.exists(dst_labels_file):
                missing_files += 1
                print(f"Warning: Missing target_labels.npy in {dst_angle_dir}")
    
    if missing_files == 0:
        print("All destination directories have target_labels.npy files.")
    else:
        print(f"Found {missing_files} missing target_labels.npy files in destination directories.")

if __name__ == "__main__":
    main() 