import logging
import torch
import os
from torch.utils.data.dataset import Dataset
import numpy as np

log = logging.getLogger(__name__)

class XRFProcessedDataset(Dataset):
    def __init__(self, base_dir='./xrf555_processed', split='train'):
        """
        Initialize the dataset.
        Args:
            base_dir (str): Base directory containing the data
            split (str): One of 'train', 'val', or 'test'
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
        
        for action_type in self.action_types:
            action_path = os.path.join(base_dir, action_type)
            ninety_path = os.path.join(action_path, '90')
            
            if not os.path.exists(ninety_path):
                print(f"Warning: No '90' folder found for action type {action_type}")
                continue
            
            # Get all numbered folders
            numbered_folders = [d for d in os.listdir(ninety_path) if os.path.isdir(os.path.join(ninety_path, d))]
            
            for folder in numbered_folders:
                folder_path = os.path.join(ninety_path, folder)
                xrf_specs_path = os.path.join(folder_path, 'xrf_specs.npy')
                xrf_sim2real_specs_path = os.path.join(folder_path, 'xrf_sim2real_specs.npy')
                
                if os.path.exists(xrf_specs_path) and os.path.exists(xrf_sim2real_specs_path):
                    all_samples.append({
                        'sim_path': xrf_specs_path,
                        'noisy_path': xrf_sim2real_specs_path,
                        'label': self.action_to_label[action_type]
                    })
        
        # Shuffle samples
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(all_samples)
        
        # Calculate split indices (70% train, 15% val, 15% test)
        total_samples = len(all_samples)
        train_idx = int(total_samples * 0.7)
        val_idx = int(total_samples * 0.85)  # 70% + 15% = 85%
        
        # Split the data according to the specified split
        if split == 'train':
            self.samples = all_samples[:train_idx]
        elif split == 'val':
            self.samples = all_samples[train_idx:val_idx]
        else:  # test
            self.samples = all_samples[val_idx:]
        
        print(f"Loaded {len(self.samples)} {split} samples from {len(self.action_types)} action types")
        print(f"Action types: {self.action_types}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load simulation data
        sim_data = np.load(sample['sim_path'])
        # Load noisy data
        noisy_data = np.load(sample['noisy_path'])
        # Get label
        label = sample['label']
        
        # Convert to PyTorch tensors
        sim_data = torch.from_numpy(sim_data).float()
        noisy_data = torch.from_numpy(noisy_data).float()
        
        return sim_data, noisy_data, label
