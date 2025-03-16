import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm

def analyze_xrf_data():
    # Base directory containing all action types
    base_dir = 'xrf555'
    
    # Get all action types (folders in the base directory)
    action_types = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Dictionary to store second dimension sizes for each action type
    action_dimensions = defaultdict(list)
    
    # Process each action type
    for action_type in action_types:
        print(f"Processing action type: {action_type}")
        action_path = os.path.join(base_dir, action_type)
        
        # Navigate to the 90 folder
        ninety_path = os.path.join(action_path, '90')
        
        if not os.path.exists(ninety_path):
            print(f"Warning: No '90' folder found for action type {action_type}")
            continue
        
        # Get all numbered folders
        numbered_folders = [d for d in os.listdir(ninety_path) if os.path.isdir(os.path.join(ninety_path, d))]
        
        # Process each numbered folder
        for folder in tqdm(numbered_folders, desc=f"Folders in {action_type}"):
            folder_path = os.path.join(ninety_path, folder)
            xrf_specs_path = os.path.join(folder_path, 'xrf_specs.npy')
            
            if os.path.exists(xrf_specs_path):
                try:
                    # Load the data and get the second dimension size
                    data = np.load(xrf_specs_path)
                    second_dim_size = data.shape[1]
                    action_dimensions[action_type].append(second_dim_size)
                except Exception as e:
                    print(f"Error loading {xrf_specs_path}: {e}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for action_type, dimensions in action_dimensions.items():
        if dimensions:
            print(f"{action_type}: Count={len(dimensions)}, Min={min(dimensions)}, Max={max(dimensions)}, Mean={np.mean(dimensions):.2f}")
    
    # Check if all actions except "wipe" have second dimension under 30
    print("\nChecking if all actions except 'wipe' have second dimension under 30:")
    for action_type, dimensions in action_dimensions.items():
        if action_type != 'wipe':
            over_30 = [d for d in dimensions if d > 30]
            if over_30:
                print(f"{action_type}: {len(over_30)}/{len(dimensions)} samples have second dimension > 30")
                print(f"  Values over 30: {sorted(over_30)}")
            else:
                print(f"{action_type}: All {len(dimensions)} samples have second dimension <= 30")
        else:
            under_30 = [d for d in dimensions if d <= 30]
            over_30 = [d for d in dimensions if d > 30]
            print(f"wipe: {len(under_30)}/{len(dimensions)} samples have second dimension <= 30")
            print(f"wipe: {len(over_30)}/{len(dimensions)} samples have second dimension > 30")
    
    # Plot distributions
    plt.figure(figsize=(15, 10))
    
    # Create a box plot for each action type
    plt.subplot(2, 1, 1)
    box_data = [dimensions for action_type, dimensions in action_dimensions.items() if dimensions]
    box_labels = [action_type for action_type, dimensions in action_dimensions.items() if dimensions]
    
    if box_data:
        plt.boxplot(box_data, labels=box_labels)
        plt.title('Distribution of Second Dimension Size by Action Type (Box Plot)')
        plt.ylabel('Second Dimension Size')
        plt.axhline(y=30, color='r', linestyle='--', label='Threshold = 30')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create histograms for each action type
    plt.subplot(2, 1, 2)
    
    # Flatten all dimensions for histogram
    all_dimensions = []
    for dimensions in action_dimensions.values():
        all_dimensions.extend(dimensions)
    
    if all_dimensions:
        sns.histplot(all_dimensions, kde=True)
        plt.title('Overall Distribution of Second Dimension Size (Histogram)')
        plt.xlabel('Second Dimension Size')
        plt.ylabel('Frequency')
        plt.axvline(x=30, color='r', linestyle='--', label='Threshold = 30')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('xrf_dimension_distribution.png')
    print("Plot saved as 'xrf_dimension_distribution.png'")
    
    # Create a separate plot for each action type
    plt.figure(figsize=(15, 10))
    
    # Determine number of subplots needed
    n_actions = len(action_dimensions)
    if n_actions > 0:
        n_cols = min(3, n_actions)
        n_rows = (n_actions + n_cols - 1) // n_cols
        
        for i, (action_type, dimensions) in enumerate(action_dimensions.items(), 1):
            if dimensions:
                plt.subplot(n_rows, n_cols, i)
                sns.histplot(dimensions, kde=True)
                plt.title(f'{action_type} (n={len(dimensions)})')
                plt.xlabel('Second Dimension Size')
                plt.ylabel('Frequency')
                plt.axvline(x=30, color='r', linestyle='--', label='Threshold = 30')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('xrf_dimension_by_action.png')
        print("Plot saved as 'xrf_dimension_by_action.png'")

if __name__ == "__main__":
    analyze_xrf_data() 