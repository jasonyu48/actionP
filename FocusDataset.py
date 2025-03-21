import logging
import torch
import os
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

log = logging.getLogger(__name__)

class FocusProcessedDataset(Dataset):
    def __init__(self, base_dir='./Focus_processed', split='train'):
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
                focus_specs_path = os.path.join(folder_path, 'focus_specs.npz')
                focus_sim2real_specs_path = os.path.join(folder_path, 'focus_sim2real_specs.npz')
                
                if os.path.exists(focus_specs_path) and os.path.exists(focus_sim2real_specs_path):
                    all_samples.append({
                        'sim_path': focus_specs_path,
                        'noisy_path': focus_sim2real_specs_path,
                        'label': self.action_to_label[action_type]
                    })
        
        # Shuffle samples
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(all_samples)
        
        # Calculate split indices (70% train, 15% val, 15% test) - 7:1.5:1.5 ratio
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
        
        # Load data from .npz files
        sim_data = np.load(sample['sim_path'])
        noisy_data = np.load(sample['noisy_path'])
        
        # Extract RDspecs and AoAspecs
        sim_RDspecs = sim_data['RDspecs']
        sim_AoAspecs = sim_data['AoAspecs']
        
        noisy_RDspecs = noisy_data['RDspecs']
        noisy_AoAspecs = noisy_data['AoAspecs']
        
        # Get label
        label = sample['label']
        
        # Convert to PyTorch tensors
        sim_RDspecs = torch.from_numpy(sim_RDspecs).float()
        sim_AoAspecs = torch.from_numpy(sim_AoAspecs).float()
        
        noisy_RDspecs = torch.from_numpy(noisy_RDspecs).float()
        noisy_AoAspecs = torch.from_numpy(noisy_AoAspecs).float()
        
        return {
            'sim_RDspecs': sim_RDspecs,
            'sim_AoAspecs': sim_AoAspecs,
            'noisy_RDspecs': noisy_RDspecs,
            'noisy_AoAspecs': noisy_AoAspecs,
            'label': label
        }

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences
    """
    # Extract components from batch
    sim_RDspecs = [item['sim_RDspecs'] for item in batch]
    sim_AoAspecs = [item['sim_AoAspecs'] for item in batch]
    noisy_RDspecs = [item['noisy_RDspecs'] for item in batch]
    noisy_AoAspecs = [item['noisy_AoAspecs'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Get lengths for padding mask
    lengths = [spec.shape[0] for spec in sim_RDspecs]
    max_len = max(lengths)
    
    # Create padding masks
    padding_masks = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_masks[i, length:] = True  # True indicates padding
    
    # Pad sequences
    padded_sim_RDspecs = pad_sequence(sim_RDspecs, batch_first=True)
    padded_sim_AoAspecs = pad_sequence(sim_AoAspecs, batch_first=True)
    padded_noisy_RDspecs = pad_sequence(noisy_RDspecs, batch_first=True)
    padded_noisy_AoAspecs = pad_sequence(noisy_AoAspecs, batch_first=True)
    labels = torch.tensor(labels)
    
    return {
        'sim_RDspecs': padded_sim_RDspecs,
        'sim_AoAspecs': padded_sim_AoAspecs,
        'noisy_RDspecs': padded_noisy_RDspecs,
        'noisy_AoAspecs': padded_noisy_AoAspecs,
        'padding_mask': padding_masks,
        'labels': labels
    }
