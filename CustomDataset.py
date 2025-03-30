import logging
import torch
import os
from torch.utils.data.dataset import Dataset
import numpy as np

log = logging.getLogger(__name__)

class XRFProcessedDataset(Dataset):
    def __init__(self, base_dir='./xrf555_processed_multi_angle', split='train', use_multi_angle=True):
        """
        Initialize the dataset.
        Args:
            base_dir (str): Base directory containing the data
            split (str): One of 'train', 'val', or 'test'
            use_multi_angle (bool): Whether to use data from all angles or just 90 degrees
        """
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        
        # Validate split parameter
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of 'train', 'val', or 'test'")
        
        # Get all action types (folders in the base directory)
        self.action_types = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        # Map action types to labels (alphabetical order for consistency)
        self.action_types.sort()
        self.action_to_label = {action: idx for idx, action in enumerate(self.action_types)}
        
        # Collect all sample paths and their corresponding labels
        all_samples = []
        # All angle folders or just 90 degrees based on the parameter
        angle_folders = ['0', '90', '180', '270'] if use_multi_angle else ['90']
        
        for action_type in self.action_types:
            action_path = os.path.join(base_dir, action_type)
            
            for angle in angle_folders:
                angle_path = os.path.join(action_path, angle)
                
                if not os.path.exists(angle_path):
                    print(f"Warning: No '{angle}' folder found for action type {action_type}")
                    continue
                
                # Get all numbered folders
                numbered_folders = [d for d in os.listdir(angle_path) if os.path.isdir(os.path.join(angle_path, d))]
                
                for folder in numbered_folders:
                    folder_path = os.path.join(angle_path, folder)
                    xrf_specs_path = os.path.join(folder_path, 'xrf_specs.npy')
                    xrf_sim2real_specs_path = os.path.join(folder_path, 'xrf_sim2real_specs.npy')
                    
                    if os.path.exists(xrf_specs_path) and os.path.exists(xrf_sim2real_specs_path):
                        all_samples.append({
                            'sim_path': xrf_specs_path,
                            'noisy_path': xrf_sim2real_specs_path,
                            'label': self.action_to_label[action_type],
                            'action_type': action_type,  # Store the action type for stratified sampling
                            'angle': angle
                        })
        
        # Use stratified sampling to ensure balanced classes in all splits
        # Group samples by action type
        samples_by_class = {}
        for action_type in self.action_types:
            samples_by_class[action_type] = [s for s in all_samples if s['action_type'] == action_type]
            # Shuffle each class's samples for randomness
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(samples_by_class[action_type])
        
        # Calculate split indices for each class (70% train, 15% val, 15% test)
        train_samples = []
        val_samples = []
        test_samples = []
        
        for action_type, samples in samples_by_class.items():
            n_samples = len(samples)
            train_idx = int(n_samples * 0.7)
            val_idx = int(n_samples * 0.85)  # 70% + 15% = 85%
            
            train_samples.extend(samples[:train_idx+1]) # +1 to make sure the number of samples is even
            val_samples.extend(samples[train_idx+1:val_idx])
            test_samples.extend(samples[val_idx:])
        
        # Shuffle the samples within each split for good measure
        np.random.seed(2025)
        np.random.shuffle(train_samples)
        np.random.shuffle(val_samples)
        np.random.shuffle(test_samples)
        
        # Assign the appropriate split
        if split == 'train':
            self.samples = train_samples
        elif split == 'val':
            self.samples = val_samples
        else:  # test
            self.samples = test_samples
        
        # Count samples from each angle
        angle_counts = {angle: sum(1 for s in self.samples if s['angle'] == angle) for angle in angle_folders}
        # Count samples from each class
        class_counts = {}
        for action_type in self.action_types:
            class_counts[action_type] = sum(1 for s in self.samples if s['action_type'] == action_type)
        
        print(f"Loaded {len(self.samples)} {split} samples from {len(self.action_types)} action types")
        print(f"  - Samples per angle: {angle_counts}")
        print(f"  - Using {'all angles' if use_multi_angle else 'only 90-degree angle'}")
        print(f"  - Samples per class:")
        for action_type, count in class_counts.items():
            print(f"      {action_type}: {count}")
        print(f"Action types: {self.action_types}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load simulation data
        sim_data = np.load(sample['sim_path'])
        # Load noisy data
        noisy_data = np.load(sample['noisy_path'])
        # Get label and angle
        label = sample['label']
        angle = int(sample['angle'])
        
        # Convert to PyTorch tensors
        sim_data = torch.from_numpy(sim_data).float()
        noisy_data = torch.from_numpy(noisy_data).float()
        
        return sim_data, noisy_data, label
