import logging
import torch
import os
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

log = logging.getLogger(__name__)

class FocusProcessedDataset(Dataset):
    """Dataset class for loading the processed Focus dataset, where the zeros are cropped around the target (peak), and the target location is provided as the last index of the last dimension"""
    def __init__(self, base_dir='./Focus_processed_multi_angle', split='train', use_multi_angle=True):
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
                    
                    # Look for both simulated and noisy data
                    sim_specs_path = os.path.join(folder_path, 'focus_specs.npz')
                    noisy_specs_path = os.path.join(folder_path, 'focus_sim2real_specs.npz')
                    
                    # Check if sim data exists (required)
                    if os.path.exists(sim_specs_path):
                        # Add noisy data path if available, else set to None
                        noisy_path = noisy_specs_path if os.path.exists(noisy_specs_path) else None
                        
                        all_samples.append({
                            'sim_path': sim_specs_path,
                            'noisy_path': noisy_path,
                            'label': self.action_to_label[action_type],
                            'action_type': action_type,  # Store the action type for stratified sampling
                            'angle': angle
                        })
                        
                        # If noisy data wasn't found, log a warning
                        if noisy_path is None:
                            print(f"Warning: No noisy data found for {folder_path}, will use sim data as fallback")
        
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
            
            train_samples.extend(samples[:train_idx])
            val_samples.extend(samples[train_idx:val_idx])
            test_samples.extend(samples[val_idx:])
        
        # Shuffle the samples within each split for good measure
        np.random.seed(42)
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
        
        # Count samples with noisy data
        noisy_count = sum(1 for s in self.samples if s['noisy_path'] is not None)
        # Count samples from each angle
        angle_counts = {angle: sum(1 for s in self.samples if s['angle'] == angle) for angle in angle_folders}
        # Count samples from each class
        class_counts = {}
        for action_type in self.action_types:
            class_counts[action_type] = sum(1 for s in self.samples if s['action_type'] == action_type)
        
        print(f"Loaded {len(self.samples)} {split} samples from {len(self.action_types)} action types")
        print(f"  - {noisy_count} samples have noisy data available")
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
        
        # Load simulated data
        sim_data = np.load(sample['sim_path'])
        
        # Extract simulated RDspecs and AoAspecs
        sim_RDspecs = sim_data['RDspecs']
        sim_AoAspecs = sim_data['AoAspecs']
        
        # Convert simulated data to PyTorch tensors
        sim_RDspecs = torch.from_numpy(sim_RDspecs).float()
        sim_AoAspecs = torch.from_numpy(sim_AoAspecs).float()
        
        # If noisy data is available, load it
        if sample['noisy_path'] is not None:
            noisy_data = np.load(sample['noisy_path'])
            noisy_RDspecs = torch.from_numpy(noisy_data['RDspecs']).float()
            noisy_AoAspecs = torch.from_numpy(noisy_data['AoAspecs']).float()
        else:
            # Fall back to using simulated data as noisy data
            noisy_RDspecs = sim_RDspecs
            noisy_AoAspecs = sim_AoAspecs
        
        # Get label and angle
        label = sample['label']
        angle = int(sample['angle'])
        
        return {
            'sim_RDspecs': sim_RDspecs,
            'sim_AoAspecs': sim_AoAspecs,
            'noisy_RDspecs': noisy_RDspecs,
            'noisy_AoAspecs': noisy_AoAspecs,
            'label': label,
            'angle': angle
        }

class FocusOriginalDataset(Dataset):
    """Dataset class for loading the dataset that is all zero except around the target (peak)"""
    def __init__(self, base_dir='/weka/scratch/rzhao36/lwang/datasets/HOI/datasets/focus', split='train', use_multi_angle=True):
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
        
        # Map action types to labels
        self.action_types.sort()  # Ensure consistent ordering
        self.action_to_label = {action: idx for idx, action in enumerate(self.action_types)}
        
        # Collect all sample paths and their corresponding labels
        all_samples = []
        
        for action_type in self.action_types:
            action_path = os.path.join(base_dir, action_type)
            
            # Get all time-stamped folders
            time_folders = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
            
            for folder in time_folders:
                folder_path = os.path.join(action_path, folder)
                raw_data_path = os.path.join(folder_path, 'raw_data_matrix.npy')
                
                if os.path.exists(raw_data_path):
                    all_samples.append({
                        'path': raw_data_path,
                        'label': self.action_to_label[action_type],
                        'action_type': action_type  # Store action type for stratified sampling
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
            
            train_samples.extend(samples[:train_idx])
            val_samples.extend(samples[train_idx:val_idx])
            test_samples.extend(samples[val_idx:])
        
        # Shuffle the samples within each split for good measure
        np.random.seed(42)
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
        
        # Count samples from each class for reporting
        class_counts = {}
        for action_type in self.action_types:
            class_counts[action_type] = sum(1 for s in self.samples if s['action_type'] == action_type)
        
        print(f"Loaded {len(self.samples)} {split} samples from {len(self.action_types)} action types")
        print(f"  - Samples per class:")
        for action_type, count in class_counts.items():
            print(f"      {action_type}: {count}")
        print(f"Action types: {self.action_types}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load data
        data = np.load(sample['path'])
        label = sample['label']
        
        # Convert to PyTorch tensor
        data = torch.from_numpy(data).float()
        
        return data, label

class LWDataset(Dataset):
    """Dataset class for loading data from the LW format dataset (Lihao's original dataset)"""
    def __init__(self, base_dir='/weka/scratch/rzhao36/lwang/datasets/HOI/datasets/classic', split='train', use_multi_angle=True):
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
                    
                    # Look for both simulated and noisy data
                    sim_specs_path = os.path.join(folder_path, 'specs.npy')
                    noisy_specs_path = os.path.join(folder_path, 'sim2real_specs.npy')
                    
                    # Check if sim data exists (required)
                    if os.path.exists(sim_specs_path):
                        # Add noisy data path if available, else set to None
                        noisy_path = noisy_specs_path if os.path.exists(noisy_specs_path) else None
                        
                        all_samples.append({
                            'sim_path': sim_specs_path,
                            'noisy_path': noisy_path,
                            'label': self.action_to_label[action_type],
                            'action_type': action_type,  # Store the action type for stratified sampling
                            'angle': angle
                        })
                        
                        # If noisy data wasn't found, log a warning
                        if noisy_path is None:
                            print(f"Warning: No noisy data found for {folder_path}, will use sim data as fallback")
        
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
            
            train_samples.extend(samples[:train_idx])
            val_samples.extend(samples[train_idx:val_idx])
            test_samples.extend(samples[val_idx:])
        
        # Shuffle the samples within each split for good measure
        np.random.seed(42)
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
        
        # Count samples with noisy data
        noisy_count = sum(1 for s in self.samples if s['noisy_path'] is not None)
        # Count samples from each angle
        angle_counts = {angle: sum(1 for s in self.samples if s['angle'] == angle) for angle in angle_folders}
        # Count samples from each class
        class_counts = {}
        for action_type in self.action_types:
            class_counts[action_type] = sum(1 for s in self.samples if s['action_type'] == action_type)
        
        print(f"Loaded {len(self.samples)} {split} samples from {len(self.action_types)} action types")
        print(f"  - {noisy_count} samples have noisy data available")
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
        
        # Load simulated data
        sim_data = np.load(sample['sim_path'])
        
        # Convert simulated data to PyTorch tensors, shape: (t, 3, 256, 256)
        sim_specs = torch.from_numpy(sim_data).float()
        
        # If noisy data is available, load it
        if sample['noisy_path'] is not None:
            noisy_data = np.load(sample['noisy_path'])
            noisy_specs = torch.from_numpy(noisy_data).float()
        else:
            # Fall back to using simulated data as noisy data
            noisy_specs = sim_specs
        
        # Get label and angle
        label = sample['label']
        angle = int(sample['angle'])
        
        return {
            'sim_specs': sim_specs,
            'noisy_specs': noisy_specs,
            'label': label,
            'angle': angle
        }

def lw_collate_fn(batch):
    """
    Custom collate function to handle variable length sequences for LWDataset
    """
    # Extract components from batch
    sim_specs = [item['sim_specs'] for item in batch]
    noisy_specs = [item['noisy_specs'] for item in batch]
    labels = [item['label'] for item in batch]
    angles = [item['angle'] for item in batch]
    
    # Get lengths for padding mask
    lengths = [spec.shape[0] for spec in sim_specs]
    max_len = max(lengths)
    
    # Create padding masks
    padding_masks = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_masks[i, length:] = True  # True indicates padding
    
    # Pad sequences
    padded_sim_specs = pad_sequence(sim_specs, batch_first=True)
    padded_noisy_specs = pad_sequence(noisy_specs, batch_first=True)
    labels = torch.tensor(labels)
    angles = torch.tensor(angles)
    
    return {
        'sim_specs': padded_sim_specs,
        'noisy_specs': padded_noisy_specs,
        'padding_mask': padding_masks,
        'labels': labels,
        'angles': angles
    }

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences
    Works with both processed and original datasets
    """
    # Extract components from batch
    sim_RDspecs = [item['sim_RDspecs'] for item in batch]
    sim_AoAspecs = [item['sim_AoAspecs'] for item in batch]
    noisy_RDspecs = [item['noisy_RDspecs'] for item in batch]
    noisy_AoAspecs = [item['noisy_AoAspecs'] for item in batch]
    labels = [item['label'] for item in batch]
    angles = [item['angle'] for item in batch]
    
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
    angles = torch.tensor(angles)
    
    return {
        'sim_RDspecs': padded_sim_RDspecs,
        'sim_AoAspecs': padded_sim_AoAspecs,
        'noisy_RDspecs': padded_noisy_RDspecs,
        'noisy_AoAspecs': padded_noisy_AoAspecs,
        'padding_mask': padding_masks,
        'labels': labels,
        'angles': angles
    }
