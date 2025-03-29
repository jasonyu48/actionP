import os
import numpy as np
from tqdm import tqdm
import shutil
import torch

def get_target_loc(doppler_spec):
    """
    Takes a radar frame and returns the target location indices
    """
    adc_samples = doppler_spec.shape[-1]
    argmax_flattened = torch.argmax(doppler_spec.view(doppler_spec.shape[0], -1), dim=-1, keepdim=True)
    vec_idx = argmax_flattened // adc_samples
    target_idx = argmax_flattened % adc_samples
    return vec_idx, target_idx

def crop_around_peak(data, target_idx, window_size=15):
    """
    Crops the data around the peak with given window size on each side
    Returns cropped data and normalized target position
    """
    # Convert target_idx to numpy if it's torch tensor
    if isinstance(target_idx, torch.Tensor):
        target_idx = target_idx.numpy()
    
    # Get the middle index for each antenna
    peak_indices = target_idx.squeeze()
    
    # Initialize output array with correct shape (num_antennas, time/velocity, 32)
    cropped_data = np.zeros((data.shape[0], data.shape[1], 31))
    norm_positions = np.zeros((data.shape[0], data.shape[1], 1))
    
    # Process each antenna separately
    for i in range(data.shape[0]):  # Loop over antennas
        peak_idx = peak_indices[i]
        
        # Calculate crop indices
        start_idx = max(0, peak_idx - window_size)
        end_idx = min(data.shape[-1], peak_idx + window_size + 1)
        
        # Pad if necessary
        pad_left = max(0, window_size - peak_idx)
        pad_right = max(0, (peak_idx + window_size + 1) - data.shape[-1])
        
        # Extract the slice for the current antenna
        slice_data = data[i, :, start_idx:end_idx]  # Shape: (time/velocity, cropped_range)
        
        # Pad if necessary - pad_width is (before_dim1, after_dim1, before_dim2, after_dim2)
        # We only pad the second dimension (range)
        pad_width = ((0, 0), (pad_left, pad_right))
        padded_data = np.pad(slice_data, pad_width, 'constant')
        
        # Assign to output
        cropped_data[i] = padded_data
        
        # Store normalized position - broadcasting to all time/velocity points
        norm_positions[i, :, 0] = peak_idx / 255.0
    
    # Concatenate the normalized position to the cropped data
    final_data = np.concatenate([cropped_data, norm_positions], axis=-1)
    return final_data

def process_focus_data():
    # Base directory containing all action types
    base_dir = '/weka/scratch/rzhao36/lwang/datasets/HOI/datasets/focus'
    output_base_dir = 'Focus_processed_multi_angle'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    # Get all action types
    action_types = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Define all angle folders to process
    angle_folders = ['0', '90', '180', '270']
    
    for action_type in action_types:
        print(f"Processing action type: {action_type}")
        action_path = os.path.join(base_dir, action_type)
        output_action_path = os.path.join(output_base_dir, action_type)
        
        if not os.path.exists(output_action_path):
            os.makedirs(output_action_path)
        
        # Process each angle folder
        for angle in angle_folders:
            angle_path = os.path.join(action_path, angle)
            output_angle_path = os.path.join(output_action_path, angle)
            
            if not os.path.exists(angle_path):
                print(f"Warning: No '{angle}' folder found for action type {action_type}")
                continue
            
            if not os.path.exists(output_angle_path):
                os.makedirs(output_angle_path)
            
            # Get all numbered folders
            numbered_folders = [d for d in os.listdir(angle_path) if os.path.isdir(os.path.join(angle_path, d))]
            
            # Process each numbered folder
            for folder in tqdm(numbered_folders, desc=f"Folders in {action_type}/{angle}"):
                folder_path = os.path.join(angle_path, folder)
                output_folder_path = os.path.join(output_angle_path, folder)
                
                if not os.path.exists(output_folder_path):
                    os.makedirs(output_folder_path)
                
                # Process focus_specs.npz and focus_sim2real_specs.npz
                for file_name in ['focus_specs.npz', 'focus_sim2real_specs.npz']:
                    input_path = os.path.join(folder_path, file_name)
                    output_path = os.path.join(output_folder_path, file_name)
                    
                    if os.path.exists(input_path):
                        try:
                            # Load the data
                            data = np.load(input_path)
                            RDspecs = data['RDspecs']
                            AoAspecs = data['AoAspecs']
                            
                            # Convert to torch tensor for processing
                            RDspecs_torch = torch.from_numpy(RDspecs)
                            
                            # Process time range and velocity range separately
                            processed_RD = np.zeros((RDspecs.shape[0], 2, RDspecs.shape[2], RDspecs.shape[3], 32))  # 32 = 31 + 1 for position
                            
                            for t in range(RDspecs.shape[0]):  # Loop over time
                                for dim in range(2):  # Loop over time/velocity dimension
                                    current_frame = RDspecs_torch[t, dim]  # Shape: (12, 128, 256)
                                    _, target_idx = get_target_loc(current_frame)  # Shape: (12, 1)
                                    # Process the frame
                                    processed_frame = crop_around_peak(current_frame.numpy(), target_idx)
                                    processed_RD[t, dim] = processed_frame
                            
                            # Save the processed data
                            np.savez(output_path, RDspecs=processed_RD, AoAspecs=AoAspecs)
                            
                        except Exception as e:
                            print(f"Error processing {input_path}: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Copy other files directly
                for file_name in os.listdir(folder_path):
                    if file_name not in ['focus_specs.npz', 'focus_sim2real_specs.npz']:
                        src_file = os.path.join(folder_path, file_name)
                        dst_file = os.path.join(output_folder_path, file_name)
                        if os.path.isfile(src_file):
                            shutil.copy2(src_file, dst_file)

if __name__ == "__main__":
    process_focus_data()