import logging
import torch
import os
from torch.utils.data.dataset import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from typing import Optional

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

class CubeLearnDataset(Dataset):
    """Dataset class for loading radar cube data for the RDAT_3DCNNLSTM (cubelearn) model"""
    def __init__(self, base_dir='/scratch/tshu2/jyu197/XRF55-repo/Focus_processed_multi_angle', split='train', use_multi_angle=True):
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
                    sim_data_path = os.path.join(folder_path, 'radar_frames.npy')
                    noisy_data_path = os.path.join(folder_path, 'focus_sim2real_radar_frames.npy')
                    
                    # Check if sim data exists (required)
                    if os.path.exists(sim_data_path):
                        # Add noisy data path if available, else set to None
                        noisy_path = noisy_data_path if os.path.exists(noisy_data_path) else None
                        
                        all_samples.append({
                            'sim_path': sim_data_path,
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
        
        # Reshape data from (radar_frames, 3, 4, 128, 256) to (radar_frames, 128, 12, 256)
        # Ensure the data is loaded as complex64
        sim_data = torch.from_numpy(sim_data.astype(np.complex64))
        batch_size, ant_dim1, ant_dim2, chirps, range_bins = sim_data.shape
        sim_data = sim_data.permute(0, 3, 1, 2, 4)  # -> (radar_frames, 128, 3, 4, 256)
        sim_data = sim_data.reshape(batch_size, chirps, ant_dim1 * ant_dim2, range_bins)  # -> (radar_frames, 128, 12, 256)
        
        # If noisy data is available, load and process it
        if sample['noisy_path'] is not None:
            noisy_data = np.load(sample['noisy_path'])
            noisy_data = torch.from_numpy(noisy_data.astype(np.complex64))
            # Apply same reshaping
            noisy_data = noisy_data.permute(0, 3, 1, 2, 4)
            noisy_data = noisy_data.reshape(batch_size, chirps, ant_dim1 * ant_dim2, range_bins)
        else:
            # Fall back to using simulated data as noisy data
            noisy_data = sim_data
        
        # Get label and angle
        label = sample['label']
        angle = int(sample['angle'])
        
        return {
            'sim_radar_frames': sim_data,
            'noisy_radar_frames': noisy_data,
            'label': label,
            'angle': angle
        }

def cubelearn_collate_fn(batch):
    """
    Custom collate function to handle variable length sequences for CubeLearnDataset
    """
    # Extract components from batch
    sim_radar_frames = [item['sim_radar_frames'] for item in batch]
    noisy_radar_frames = [item['noisy_radar_frames'] for item in batch]
    labels = [item['label'] for item in batch]
    angles = [item['angle'] for item in batch]
    
    # Get lengths for padding mask
    lengths = [frames.shape[0] for frames in sim_radar_frames]
    max_len = max(lengths)
    
    # Create padding masks
    padding_masks = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_masks[i, length:] = True  # True indicates padding
    
    # Pad sequences
    padded_sim_radar_frames = pad_sequence(sim_radar_frames, batch_first=True)
    padded_noisy_radar_frames = pad_sequence(noisy_radar_frames, batch_first=True)
    labels = torch.tensor(labels)
    angles = torch.tensor(angles)
    
    return {
        'sim_radar_frames': padded_sim_radar_frames,
        'noisy_radar_frames': padded_noisy_radar_frames,
        'padding_mask': padding_masks,
        'labels': labels,
        'angles': angles
    }

class FocusDatasetwithRFID(Dataset):
    """Dataset class for loading both radar and RFID data for human-object interaction recognition"""
    def __init__(self, base_dir='/scratch/tshu2/jyu197/Focus_processed_multi_angle_rfid', split='train', use_multi_angle=True, use_noisy=True,
                 # New splitting strategy parameters:
                 split_strategy='default',  # 'default', 'angle-based', 'random-subset'
                 train_angles=None,  # for angle-based, list of angles to use for training, e.g. ['0', '90']
                 val_angle=None,     # for angle-based, angle to use for validation, e.g. '180'
                 samples_per_class=None,  # for angle-based or random-subset, limit samples per class
                 ):
        """
        Initialize the dataset with both radar and RFID data.
        Args:
            base_dir (str): Base directory containing the data
            split (str): One of 'train', 'val', or 'test'
            use_multi_angle (bool): Whether to use data from all angles or just 90 degrees
            use_noisy (bool): Whether to use noisy data (when available) or clean data only
            split_strategy (str): How to split data: 'default' (70/15/15), 'angle-based' (specific angles for train/val), 
                                 or 'random-subset' (random n samples per class)
            train_angles (list): For angle-based strategy, which angles to use for training (e.g. ['0', '90'])
            val_angle (str): For angle-based strategy, which angle to use for validation (e.g. '180')
            samples_per_class (int): For angle-based or random-subset, maximum samples per action class
        """
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.use_noisy = use_noisy
        self.split_strategy = split_strategy
        self.train_angles = train_angles
        self.val_angle = val_angle
        self.samples_per_class = samples_per_class
        
        # Define the list of possible objects and their mapping to indices
        self.object_names = ['bottle', 'pen', 'microwave', 'cabinet', 'phone', 'chair', 'book', 'table']
        self.object_to_idx = {obj: idx for idx, obj in enumerate(self.object_names)}
        
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
        if split_strategy == 'angle-based' and (train_angles is not None or val_angle is not None):
            if split == 'train' and train_angles is not None:
                angle_folders = train_angles
            elif split == 'val' and val_angle is not None:
                angle_folders = [val_angle]
            else:
                # Default to all angles for test or if specific angles not provided
                angle_folders = ['0', '90', '180', '270'] if use_multi_angle else ['90']
        else:
            angle_folders = ['0', '90', '180', '270'] if use_multi_angle else ['90']
        
        # Load all target_labels.npy files first
        self.object_labels = {}
        for action_type in self.action_types:
            action_path = os.path.join(base_dir, action_type)
            
            for angle in angle_folders:
                angle_path = os.path.join(action_path, angle)
                
                if not os.path.exists(angle_path):
                    print(f"Warning: No '{angle}' folder found for action type {action_type}")
                    continue
                
                # Load target_labels.npy (containing object labels 0-5)
                target_labels_path = os.path.join(angle_path, 'target_labels.npy')
                if os.path.exists(target_labels_path):
                    try:
                        self.object_labels[f"{action_type}_{angle}"] = np.load(target_labels_path)
                        print(f"Loaded {len(self.object_labels[f'{action_type}_{angle}'])} object labels for {action_type}_{angle}")
                    except Exception as e:
                        print(f"Error loading {target_labels_path}: {e}")
                        self.object_labels[f"{action_type}_{angle}"] = None
                else:
                    print(f"Warning: No target_labels.npy found for {action_type}_{angle}")
                    self.object_labels[f"{action_type}_{angle}"] = None
        
        # Now collect all samples
        for action_type in self.action_types:
            action_path = os.path.join(base_dir, action_type)
            
            for angle in angle_folders:
                angle_path = os.path.join(action_path, angle)
                
                if not os.path.exists(angle_path):
                    continue
                
                # Skip if no object labels available for this action/angle
                if self.object_labels[f"{action_type}_{angle}"] is None:
                    print(f"Skipping {action_type}_{angle}: No object labels available")
                    continue
                
                # Get all numbered folders
                numbered_folders = [d for d in os.listdir(angle_path) if os.path.isdir(os.path.join(angle_path, d))]
                
                # Sort folders to ensure consistent indexing with target_labels.npy
                try:
                    numbered_folders = sorted(numbered_folders, key=int)
                except ValueError:
                    print(f"Warning: Folders in {angle_path} are not numeric, using lexicographic sorting")
                    numbered_folders.sort()
                
                for i, folder in enumerate(numbered_folders):
                    folder_path = os.path.join(angle_path, folder)
                    
                    # Skip if folder index exceeds available object labels
                    if i >= len(self.object_labels[f"{action_type}_{angle}"]):
                        print(f"Skipping {folder_path}: Index {i} exceeds available object labels")
                        continue
                    
                    # Look for radar data files
                    sim_specs_path = os.path.join(folder_path, 'focus_specs.npz')
                    noisy_specs_path = os.path.join(folder_path, 'focus_sim2real_specs.npz')
                    
                    # Look for RFID data files
                    rfid_clean_path = os.path.join(folder_path, 'obj_clean_phase.npy')
                    rfid_noisy_path = os.path.join(folder_path, 'obj_refined_phase.npy')
                    
                    # Look for object names file
                    obj_names_path = os.path.join(folder_path, 'obj_names.npy')
                    
                    # Check if essential data exists
                    if os.path.exists(sim_specs_path) and os.path.exists(rfid_clean_path) and os.path.exists(obj_names_path):
                        # For noisy data, fall back to clean if not available
                        noisy_radar_path = noisy_specs_path if os.path.exists(noisy_specs_path) else None
                        noisy_rfid_path = rfid_noisy_path if os.path.exists(rfid_noisy_path) else None
                        
                        # Get object label from the corresponding position in target_labels.npy
                        obj_label = int(self.object_labels[f"{action_type}_{angle}"][i])
                        
                        all_samples.append({
                            'sim_radar_path': sim_specs_path,
                            'noisy_radar_path': noisy_radar_path,
                            'clean_rfid_path': rfid_clean_path,
                            'noisy_rfid_path': noisy_rfid_path,
                            'obj_names_path': obj_names_path,
                            'action_label': self.action_to_label[action_type],
                            'obj_label': obj_label,
                            'action_type': action_type,
                            'angle': angle,
                            'folder_idx': i  # Store folder index for debugging
                        })
                        
                        # Warning for missing noisy data
                        if noisy_radar_path is None:
                            print(f"Warning: No noisy radar data found for {folder_path}")
                        if noisy_rfid_path is None:
                            print(f"Warning: No noisy RFID data found for {folder_path}")
                    else:
                        missing = []
                        if not os.path.exists(sim_specs_path): missing.append("radar data")
                        if not os.path.exists(rfid_clean_path): missing.append("RFID data")
                        if not os.path.exists(obj_names_path): missing.append("object names data")
                        print(f"Skipping {folder_path}: missing {', '.join(missing)}")
        
        # Apply splitting strategy
        np.random.seed(42)  # For reproducibility
        
        if split_strategy == 'default':
            # Use stratified sampling to ensure balanced classes in all splits (70/15/15)
            # Group samples by action type
            samples_by_class = {}
            for action_type in self.action_types:
                samples_by_class[action_type] = [s for s in all_samples if s['action_type'] == action_type]
                # Shuffle each class's samples for randomness
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
                
        elif split_strategy == 'angle-based':
            # All samples are already filtered by angle in the collection phase
            # but we still need to limit samples per class if requested
            if self.samples_per_class is not None and self.samples_per_class > 0:
                # Group samples by action type
                samples_by_class = {}
                for action_type in self.action_types:
                    samples_by_class[action_type] = [s for s in all_samples if s['action_type'] == action_type]
                    np.random.shuffle(samples_by_class[action_type])
                    # Limit samples per class
                    samples_by_class[action_type] = samples_by_class[action_type][:self.samples_per_class]
                
                # Combine limited samples
                limited_samples = []
                for samples in samples_by_class.values():
                    limited_samples.extend(samples)
                
                self.samples = limited_samples
            else:
                self.samples = all_samples
                
        elif split_strategy == 'random-subset':
            # Group samples by action type
            samples_by_class = {}
            for action_type in self.action_types:
                samples_by_class[action_type] = [s for s in all_samples if s['action_type'] == action_type]
                # Shuffle each class's samples for randomness
                np.random.shuffle(samples_by_class[action_type])
            
            # For training, take the first N samples per class
            # For validation, take the next M samples
            # For test, take the remaining samples
            train_samples = []
            val_samples = []
            test_samples = []
            
            for action_type, samples in samples_by_class.items():
                if self.samples_per_class is not None and self.samples_per_class > 0:
                    max_train = min(self.samples_per_class, len(samples) // 2)  # Ensure we leave some for val/test
                    max_val = min(self.samples_per_class // 2, (len(samples) - max_train) // 2)  # Half as many for val
                    
                    train_samples.extend(samples[:max_train])
                    val_samples.extend(samples[max_train:max_train + max_val])
                    test_samples.extend(samples[max_train + max_val:])
                else:
                    # Default behavior without sample limit
                    n_train = len(samples) // 2
                    n_val = len(samples) // 4
                    
                    train_samples.extend(samples[:n_train])
                    val_samples.extend(samples[n_train:n_train + n_val])
                    test_samples.extend(samples[n_train + n_val:])
            
            # Shuffle the samples within each split
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
        else:
            raise ValueError("split_strategy must be one of 'default', 'angle-based', or 'random-subset'")
        
        # Count samples with noisy data
        noisy_radar_count = sum(1 for s in self.samples if s['noisy_radar_path'] is not None)
        noisy_rfid_count = sum(1 for s in self.samples if s['noisy_rfid_path'] is not None)
        # Count samples from each angle
        angle_counts = {angle: sum(1 for s in self.samples if s['angle'] == angle) for angle in angle_folders}
        # Count samples from each class
        class_counts = {}
        for action_type in self.action_types:
            class_counts[action_type] = sum(1 for s in self.samples if s['action_type'] == action_type)
        # Count samples for each object
        obj_counts = {}
        for i in range(6):  # Assuming 6 object classes (0-5)
            obj_counts[i] = sum(1 for s in self.samples if s['obj_label'] == i)
        
        print(f"Loaded {len(self.samples)} {split} samples from {len(self.action_types)} action types")
        print(f"  - Split strategy: {split_strategy}")
        if split_strategy == 'angle-based':
            print(f"    - Train angles: {train_angles}")
            print(f"    - Val angle: {val_angle}")
        if samples_per_class is not None:
            print(f"    - Max samples per class: {samples_per_class}")
        print(f"  - {noisy_radar_count} samples have noisy radar data available")
        print(f"  - {noisy_rfid_count} samples have noisy RFID data available")
        print(f"  - Samples per angle: {angle_counts}")
        print(f"  - Using {'all angles' if use_multi_angle else 'only 90-degree angle'}")
        print(f"  - Using {'noisy data when available' if use_noisy else 'clean data only'}")
        print(f"  - Samples per action class:")
        for action_type, count in class_counts.items():
            print(f"      {action_type}: {count}")
        print(f"  - Samples per object class:")
        for obj_id, count in obj_counts.items():
            print(f"      Object {obj_id}: {count}")
        print(f"Action types: {self.action_types}")
        print(f"Object names: {self.object_names}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load radar data
        sim_radar_data = np.load(sample['sim_radar_path'])
        
        # Extract simulated RDspecs and AoAspecs
        sim_RDspecs = sim_radar_data['RDspecs']
        sim_AoAspecs = sim_radar_data['AoAspecs']
        
        # Convert simulated data to PyTorch tensors
        sim_RDspecs = torch.from_numpy(sim_RDspecs).float()
        sim_AoAspecs = torch.from_numpy(sim_AoAspecs).float()
        
        # Load noisy radar data if available and requested
        if self.use_noisy and sample['noisy_radar_path'] is not None:
            noisy_radar_data = np.load(sample['noisy_radar_path'])
            noisy_RDspecs = torch.from_numpy(noisy_radar_data['RDspecs']).float()
            noisy_AoAspecs = torch.from_numpy(noisy_radar_data['AoAspecs']).float()
        else:
            # Fall back to using simulated data
            noisy_RDspecs = sim_RDspecs
            noisy_AoAspecs = sim_AoAspecs
        
        # Load RFID data
        clean_rfid_data = np.load(sample['clean_rfid_path'])
        
        # Load noisy RFID data if available and requested
        if self.use_noisy and sample['noisy_rfid_path'] is not None:
            noisy_rfid_data = np.load(sample['noisy_rfid_path'])
        else:
            # Fall back to using clean data
            noisy_rfid_data = clean_rfid_data
        
        # Load object names data and create one-hot encodings
        obj_names = np.load(sample['obj_names_path'], allow_pickle=True)
        
        # Create one-hot encodings for each object
        obj_one_hot_encodings = []
        for obj_name in obj_names:
            # Create a one-hot vector for this object
            one_hot = np.zeros(len(self.object_names))
            if obj_name in self.object_to_idx:
                one_hot[self.object_to_idx[obj_name]] = 1.0
            obj_one_hot_encodings.append(one_hot)
        
        # Convert to tensor and reshape to match RFID data format
        obj_one_hot_encodings = np.array(obj_one_hot_encodings)
        obj_one_hot_tensor = torch.from_numpy(obj_one_hot_encodings).float()
        
        # Expand the one-hot encodings to match the time dimension of RFID data
        time_steps = clean_rfid_data.shape[0]
        expanded_obj_one_hot = obj_one_hot_tensor.unsqueeze(0).expand(time_steps, -1, -1)  # (time_steps, 6, 8)
        
        # Convert RFID data to PyTorch tensors
        clean_rfid_data = torch.from_numpy(clean_rfid_data).float()
        noisy_rfid_data = torch.from_numpy(noisy_rfid_data).float()
        
        # Concatenate one-hot encodings with RFID data along the last dimension
        # Assuming clean_rfid_data and noisy_rfid_data have shape (time_steps, 6, 4)
        # Add the one-hot encodings to create (time_steps, 6, 4+8)
        clean_rfid_with_obj = torch.cat([clean_rfid_data, expanded_obj_one_hot], dim=-1)
        noisy_rfid_with_obj = torch.cat([noisy_rfid_data, expanded_obj_one_hot], dim=-1)
        
        # Get labels
        action_label = sample['action_label']
        obj_label = sample['obj_label']
        
        # Get angle
        angle = int(sample['angle'])
        
        return {
            'sim_RDspecs': sim_RDspecs,  # Radar data
            'sim_AoAspecs': sim_AoAspecs,
            'noisy_RDspecs': noisy_RDspecs,
            'noisy_AoAspecs': noisy_AoAspecs,
            'clean_rfid_data': clean_rfid_with_obj,  # RFID data with object one-hot encodings
            'noisy_rfid_data': noisy_rfid_with_obj,
            'action_label': action_label,  # Labels
            'obj_label': obj_label,
            'angle': angle
        }

def rfid_collate_fn(batch):
    """
    Custom collate function to handle variable length sequences for both radar and RFID data
    """
    # Extract components from batch - Radar data
    sim_RDspecs = [item['sim_RDspecs'] for item in batch]
    sim_AoAspecs = [item['sim_AoAspecs'] for item in batch]
    noisy_RDspecs = [item['noisy_RDspecs'] for item in batch]
    noisy_AoAspecs = [item['noisy_AoAspecs'] for item in batch]
    
    # Extract components from batch - RFID data (now including one-hot encoding of object names)
    clean_rfid_data = [item['clean_rfid_data'] for item in batch]
    noisy_rfid_data = [item['noisy_rfid_data'] for item in batch]
    
    # Extract components from batch - Labels and angle
    action_labels = [item['action_label'] for item in batch]
    obj_labels = [item['obj_label'] for item in batch]
    angles = [item['angle'] for item in batch]
    
    # Get lengths for padding masks - Radar
    radar_lengths = [spec.shape[0] for spec in sim_RDspecs]
    radar_max_len = max(radar_lengths)
    
    # Get lengths for padding masks - RFID
    rfid_lengths = [data.shape[0] for data in clean_rfid_data]
    rfid_max_len = max(rfid_lengths)
    
    # Create padding masks - Radar
    radar_padding_masks = torch.zeros((len(batch), radar_max_len), dtype=torch.bool)
    for i, length in enumerate(radar_lengths):
        radar_padding_masks[i, length:] = True  # True indicates padding
    
    # Create padding masks - RFID
    rfid_padding_masks = torch.zeros((len(batch), rfid_max_len), dtype=torch.bool)
    for i, length in enumerate(rfid_lengths):
        rfid_padding_masks[i, length:] = True  # True indicates padding
    
    # Pad sequences - Radar
    padded_sim_RDspecs = pad_sequence(sim_RDspecs, batch_first=True)
    padded_sim_AoAspecs = pad_sequence(sim_AoAspecs, batch_first=True)
    padded_noisy_RDspecs = pad_sequence(noisy_RDspecs, batch_first=True)
    padded_noisy_AoAspecs = pad_sequence(noisy_AoAspecs, batch_first=True)
    
    # Pad sequences - RFID (with object one-hot encodings included)
    padded_clean_rfid_data = pad_sequence(clean_rfid_data, batch_first=True)
    padded_noisy_rfid_data = pad_sequence(noisy_rfid_data, batch_first=True)
    
    # Convert labels to tensors
    action_labels = torch.tensor(action_labels)
    obj_labels = torch.tensor(obj_labels)
    angles = torch.tensor(angles)
    
    return {
        # Radar data with mask
        'radar_data': [padded_sim_RDspecs, padded_sim_AoAspecs, radar_padding_masks],
        'noisy_radar_data': [padded_noisy_RDspecs, padded_noisy_AoAspecs, radar_padding_masks],
        
        # RFID data with mask (now includes object name one-hot encodings)
        'rfid_data': [padded_clean_rfid_data, rfid_padding_masks],
        'noisy_rfid_data': [padded_noisy_rfid_data, rfid_padding_masks],
        
        # Labels
        'action_labels': action_labels,
        'obj_labels': obj_labels,
        'angles': angles
    }

class FocusRealDatasetwithRFID(Dataset):
    """Dataset class for loading real-world radar-RFID data for human-object interaction recognition.

    Two splitting strategies are supported:
    1. Random-subset split (default):
       From every action class, ``random_subset_n`` samples are drawn uniformly at random (seed=42) to
       constitute the training split.  The remaining samples form the validation split.
    2. Angle-based split:
       If ``val_angle`` (e.g. "90") is provided, all samples whose ``angle`` equals ``val_angle`` are used
       for validation, while the remaining samples are assigned to training.
    """
    def __init__(
        self,
        base_dir: str = '/scratch/tshu2/jyu197/Focus_processed_multi_angle_rfid_real',
        split: str = 'train',
        use_multi_angle: bool = True,
        random_subset_n: int = 37,
        val_angle: Optional[str] = None,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.random_subset_n = random_subset_n
        self.val_angle = val_angle  # if not None we use angle-based splitting

        # Supported splits
        if split not in ['train', 'val']:
            raise ValueError("split must be either 'train' or 'val'")

        # Define the list of possible objects and their mapping to indices (same as synthetic dataset)
        self.object_names = ['bottle', 'pen', 'microwave', 'cabinet', 'phone', 'chair', 'book', 'table']
        self.object_to_idx = {obj: idx for idx, obj in enumerate(self.object_names)}

        # Discover all action types available in the real-world dataset
        self.action_types = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        self.action_types.sort()  # keep deterministic ordering
        self.action_to_label = {action: idx for idx, action in enumerate(self.action_types)}

        # Decide which angles to iterate over
        angle_folders = ['0', '90', '180', '270'] if use_multi_angle else ['90']

        # ------------------------------------------------------------------
        # First pass – load the per-action/angle target_labels.npy files
        # ------------------------------------------------------------------
        self.object_labels: dict[str, np.ndarray | None] = {}
        for action_type in self.action_types:
            for angle in angle_folders:
                angle_path = os.path.join(base_dir, action_type, angle)
                if not os.path.exists(angle_path):
                    print(f"Warning: No '{angle}' folder found for action type {action_type}")
                    self.object_labels[f"{action_type}_{angle}"] = None
                    continue

                target_labels_path = os.path.join(angle_path, 'target_labels.npy')
                if os.path.exists(target_labels_path):
                    try:
                        self.object_labels[f"{action_type}_{angle}"] = np.load(target_labels_path)
                        print(f"Loaded {len(self.object_labels[f'{action_type}_{angle}'])} object labels for {action_type}_{angle}")
                    except Exception as e:
                        print(f"Error loading {target_labels_path}: {e}")
                        self.object_labels[f"{action_type}_{angle}"] = None
                else:
                    print(f"Warning: No target_labels.npy found for {action_type}_{angle}")
                    self.object_labels[f"{action_type}_{angle}"] = None

        # ------------------------------------------------------------------
        # Second pass – enumerate every valid sample folder
        # ------------------------------------------------------------------
        all_samples: list[dict] = []
        for action_type in self.action_types:
            for angle in angle_folders:
                angle_path = os.path.join(base_dir, action_type, angle)
                if not os.path.exists(angle_path):
                    continue

                # Skip if we failed to load object labels for this (action, angle)
                if self.object_labels.get(f"{action_type}_{angle}") is None:
                    print(f"Skipping {action_type}_{angle}: No object labels available")
                    continue

                numbered_folders = [d for d in os.listdir(angle_path) if os.path.isdir(os.path.join(angle_path, d))]
                try:
                    numbered_folders = sorted(numbered_folders, key=int)
                except ValueError:
                    print(f"Warning: Folders in {angle_path} are not numeric – using lexicographic sort")
                    numbered_folders.sort()

                for idx_in_angle, folder in enumerate(numbered_folders):
                    if idx_in_angle >= len(self.object_labels[f"{action_type}_{angle}"]):
                        print(
                            f"Skipping {os.path.join(angle_path, folder)}: Index {idx_in_angle} exceeds available object labels"
                        )
                        continue

                    folder_path = os.path.join(angle_path, folder)

                    # Determine radar and RFID file paths.  The naming convention for real-world data is assumed to be
                    # identical to the synthetic dataset for maximum compatibility.  We fall back between alternatives.
                    radar_path = os.path.join(folder_path, 'focus_specs.npz') if os.path.exists(os.path.join(folder_path, 'focus_specs.npz')) else None

                    # Real-world RFID phases file
                    rfid_path = os.path.join(folder_path, 'rfid_phases.npy') if os.path.exists(os.path.join(folder_path, 'rfid_phases.npy')) else None

                    # The dataset also contains raw radar cubes (radar_frames.npy). We ignore them here but keep the
                    # path in case future extensions need them.
                    radar_cube_path = os.path.join(folder_path, 'radar_frames.npy') if os.path.exists(os.path.join(folder_path, 'radar_frames.npy')) else None
                    # (The cube path is stored but not loaded to keep memory usage low.)
                    if radar_cube_path:
                        pass  # placeholder for potential future use

                    obj_names_path = os.path.join(folder_path, 'obj_names.npy')

                    if radar_path and rfid_path and os.path.exists(obj_names_path):
                        all_samples.append(
                            {
                                'radar_path': radar_path,
                                'rfid_path': rfid_path,
                                'obj_names_path': obj_names_path,
                                'action_label': self.action_to_label[action_type],
                                'obj_label': int(self.object_labels[f"{action_type}_{angle}"][idx_in_angle]),
                                'action_type': action_type,
                                'angle': angle,
                                'folder_idx': idx_in_angle,
                            }
                        )
                    else:
                        missing = []
                        if not radar_path:
                            missing.append('radar npz')
                        if not rfid_path:
                            missing.append('RFID npy')
                        if not os.path.exists(obj_names_path):
                            missing.append('obj_names.npy')
                        print(f"Skipping {folder_path}: missing {', '.join(missing)}")

        # ------------------------------------------------------------------
        # Construct train/val splits according to the chosen strategy
        # ------------------------------------------------------------------
        np.random.seed(42)  # deterministic shuffling
        if self.val_angle is not None:
            # Angle-based split
            train_samples = [s for s in all_samples if s['angle'] != self.val_angle]
            val_samples = [s for s in all_samples if s['angle'] == self.val_angle]
        else:
            # Random-subset split per action_type
            samples_by_class: dict[str, list] = {}
            for action_type in self.action_types:
                samples_by_class[action_type] = [s for s in all_samples if s['action_type'] == action_type]
                np.random.shuffle(samples_by_class[action_type])

            train_samples, val_samples = [], []
            for action_type, class_samples in samples_by_class.items():
                n_train = min(self.random_subset_n, len(class_samples))
                train_samples.extend(class_samples[:n_train])
                val_samples.extend(class_samples[n_train:])

        # Final assignment based on requested split
        if split == 'train':
            self.samples = train_samples
        else:  # 'val'
            self.samples = val_samples

        # ------------------------------------------------------------------
        # Diagnostic printing
        # ------------------------------------------------------------------
        angle_counts = {angle: sum(1 for s in self.samples if s['angle'] == angle) for angle in angle_folders}
        class_counts = {action: sum(1 for s in self.samples if s['action_type'] == action) for action in self.action_types}
        obj_counts = {i: sum(1 for s in self.samples if s['obj_label'] == i) for i in range(len(self.object_names))}

        print(
            f"Loaded {len(self.samples)} {split} samples from {len(self.action_types)} action types (real-world dataset)"
        )
        print(f"  - Split strategy: {'angle='+self.val_angle if self.val_angle else f'random_subset_n={self.random_subset_n}'}")
        print(f"  - Samples per angle: {angle_counts}")
        print(f"  - Samples per action class:")
        for action, cnt in class_counts.items():
            print(f"      {action}: {cnt}")
        print(f"  - Samples per object class:")
        for obj_id, cnt in obj_counts.items():
            print(f"      Object {obj_id}: {cnt}")
        print(f"Action types: {self.action_types}")
        print(f"Object names: {self.object_names}")

    # ----------------------------------------------------------------------
    # PyTorch Dataset protocol
    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ---------------------------- Radar ----------------------------
        radar_npz = np.load(sample['radar_path'])
        sim_RDspecs = torch.from_numpy(radar_npz['RDspecs']).float()
        sim_AoAspecs = torch.from_numpy(radar_npz['AoAspecs']).float()

        # For compatibility with rfid_collate_fn, we expose the real data twice
        noisy_RDspecs = sim_RDspecs
        noisy_AoAspecs = sim_AoAspecs

        # ----------------------------- RFID ----------------------------
        rfid_data_np = np.load(sample['rfid_path'])
        clean_rfid_tensor = torch.from_numpy(rfid_data_np).float()
        noisy_rfid_tensor = clean_rfid_tensor  # identical for real-world dataset

        # Object one-hot encoding
        obj_names = np.load(sample['obj_names_path'], allow_pickle=True)
        one_hots = np.zeros((len(obj_names), len(self.object_names)), dtype=np.float32)
        for i, obj_name in enumerate(obj_names):
            if obj_name in self.object_to_idx:
                one_hots[i, self.object_to_idx[obj_name]] = 1.0
        one_hots = torch.from_numpy(one_hots)  # (N_tags, 8)

        time_steps = clean_rfid_tensor.shape[0]
        expanded_one_hots = one_hots.unsqueeze(0).expand(time_steps, -1, -1)  # (T, N_tags, 8)

        clean_rfid_with_obj = torch.cat([clean_rfid_tensor, expanded_one_hots], dim=-1)
        noisy_rfid_with_obj = clean_rfid_with_obj  # identical

        return {
            'sim_RDspecs': sim_RDspecs,
            'sim_AoAspecs': sim_AoAspecs,
            'noisy_RDspecs': noisy_RDspecs,
            'noisy_AoAspecs': noisy_AoAspecs,
            'clean_rfid_data': clean_rfid_with_obj,
            'noisy_rfid_data': noisy_rfid_with_obj,
            'action_label': sample['action_label'],
            'obj_label': sample['obj_label'],
            'angle': int(sample['angle']),
        }


class DatasetwithRFID(Dataset):
    """Dataset class for loading both radar and RFID data for human-object interaction recognition"""
    def __init__(self, base_dir='/scratch/tshu2/jyu197/Focus_processed_multi_angle_rfid', split='train', use_multi_angle=True, use_noisy=True,
                 # New splitting strategy parameters:
                 split_strategy='default',  # 'default', 'angle-based', 'random-subset'
                 train_angles=None,  # for angle-based, list of angles to use for training, e.g. ['0', '90']
                 val_angle=None,     # for angle-based, angle to use for validation, e.g. '180'
                 samples_per_class=None,  # for angle-based or random-subset, limit samples per class
                 ):
        """
        Initialize the dataset with both radar and RFID data.
        Args:
            base_dir (str): Base directory containing the data
            split (str): One of 'train', 'val', or 'test'
            use_multi_angle (bool): Whether to use data from all angles or just 90 degrees
            use_noisy (bool): Whether to use noisy data (when available) or clean data only
            split_strategy (str): How to split data: 'default' (70/15/15), 'angle-based' (specific angles for train/val), 
                                 or 'random-subset' (random n samples per class)
            train_angles (list): For angle-based strategy, which angles to use for training (e.g. ['0', '90'])
            val_angle (str): For angle-based strategy, which angle to use for validation (e.g. '180')
            samples_per_class (int): For angle-based or random-subset, maximum samples per action class
        """
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.use_noisy = use_noisy
        self.split_strategy = split_strategy
        self.train_angles = train_angles
        self.val_angle = val_angle
        self.samples_per_class = samples_per_class
        
        # Define the list of possible objects and their mapping to indices
        self.object_names = ['bottle', 'pen', 'microwave', 'cabinet', 'phone', 'chair', 'book', 'table']
        self.object_to_idx = {obj: idx for idx, obj in enumerate(self.object_names)}
        
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
        if split_strategy == 'angle-based' and (train_angles is not None or val_angle is not None):
            if split == 'train' and train_angles is not None:
                angle_folders = train_angles
            elif split == 'val' and val_angle is not None:
                angle_folders = [val_angle]
            else:
                # Default to all angles for test or if specific angles not provided
                angle_folders = ['0', '90', '180', '270'] if use_multi_angle else ['90']
        else:
            angle_folders = ['0', '90', '180', '270'] if use_multi_angle else ['90']
        
        # Load all target_labels.npy files first
        self.object_labels = {}
        for action_type in self.action_types:
            action_path = os.path.join(base_dir, action_type)
            
            for angle in angle_folders:
                angle_path = os.path.join(action_path, angle)
                
                if not os.path.exists(angle_path):
                    print(f"Warning: No '{angle}' folder found for action type {action_type}")
                    continue
                
                # Load target_labels.npy (containing object labels 0-5)
                target_labels_path = os.path.join(angle_path, 'target_labels.npy')
                if os.path.exists(target_labels_path):
                    try:
                        self.object_labels[f"{action_type}_{angle}"] = np.load(target_labels_path)
                        print(f"Loaded {len(self.object_labels[f'{action_type}_{angle}'])} object labels for {action_type}_{angle}")
                    except Exception as e:
                        print(f"Error loading {target_labels_path}: {e}")
                        self.object_labels[f"{action_type}_{angle}"] = None
                else:
                    print(f"Warning: No target_labels.npy found for {action_type}_{angle}")
                    self.object_labels[f"{action_type}_{angle}"] = None
        
        # Now collect all samples
        for action_type in self.action_types:
            action_path = os.path.join(base_dir, action_type)
            
            for angle in angle_folders:
                angle_path = os.path.join(action_path, angle)
                
                if not os.path.exists(angle_path):
                    continue
                
                # Skip if no object labels available for this action/angle
                if self.object_labels[f"{action_type}_{angle}"] is None:
                    print(f"Skipping {action_type}_{angle}: No object labels available")
                    continue
                
                # Get all numbered folders
                numbered_folders = [d for d in os.listdir(angle_path) if os.path.isdir(os.path.join(angle_path, d))]
                
                # Sort folders to ensure consistent indexing with target_labels.npy
                try:
                    numbered_folders = sorted(numbered_folders, key=int)
                except ValueError:
                    print(f"Warning: Folders in {angle_path} are not numeric, using lexicographic sorting")
                    numbered_folders.sort()
                
                for i, folder in enumerate(numbered_folders):
                    folder_path = os.path.join(angle_path, folder)
                    
                    # Skip if folder index exceeds available object labels
                    if i >= len(self.object_labels[f"{action_type}_{angle}"]):
                        print(f"Skipping {folder_path}: Index {i} exceeds available object labels")
                        continue
                    
                    # Look for radar data files - CHANGED FOR CLASSIC FORMAT
                    specs_path = os.path.join(folder_path, 'specs.npy')
                    noisy_specs_path = os.path.join(folder_path, 'sim2real_specs.npy')
                    
                    # Look for RFID data files
                    rfid_clean_path = os.path.join(folder_path, 'obj_clean_phase.npy')
                    rfid_noisy_path = os.path.join(folder_path, 'obj_refined_phase.npy')
                    
                    # Look for object names file
                    obj_names_path = os.path.join(folder_path, 'obj_names.npy')
                    
                    # Check if essential data exists
                    if os.path.exists(specs_path) and os.path.exists(rfid_clean_path) and os.path.exists(obj_names_path):
                        # For noisy data, fall back to clean if not available
                        noisy_radar_path = noisy_specs_path if os.path.exists(noisy_specs_path) else None
                        noisy_rfid_path = rfid_noisy_path if os.path.exists(rfid_noisy_path) else None
                        
                        # Get object label from the corresponding position in target_labels.npy
                        obj_label = int(self.object_labels[f"{action_type}_{angle}"][i])
                        
                        all_samples.append({
                            'sim_radar_path': specs_path,
                            'noisy_radar_path': noisy_radar_path,
                            'clean_rfid_path': rfid_clean_path,
                            'noisy_rfid_path': noisy_rfid_path,
                            'obj_names_path': obj_names_path,
                            'action_label': self.action_to_label[action_type],
                            'obj_label': obj_label,
                            'action_type': action_type,
                            'angle': angle,
                            'folder_idx': i  # Store folder index for debugging
                        })
                        
                        # Warning for missing noisy data
                        if noisy_radar_path is None:
                            print(f"Warning: No noisy radar data found for {folder_path}")
                        if noisy_rfid_path is None:
                            print(f"Warning: No noisy RFID data found for {folder_path}")
                    else:
                        missing = []
                        if not os.path.exists(specs_path): missing.append("radar data")
                        if not os.path.exists(rfid_clean_path): missing.append("RFID data")
                        if not os.path.exists(obj_names_path): missing.append("object names data")
                        print(f"Skipping {folder_path}: missing {', '.join(missing)}")
        
        # Apply splitting strategy
        np.random.seed(42)  # For reproducibility
        
        if split_strategy == 'default':
            # Use stratified sampling to ensure balanced classes in all splits (70/15/15)
            # Group samples by action type
            samples_by_class = {}
            for action_type in self.action_types:
                samples_by_class[action_type] = [s for s in all_samples if s['action_type'] == action_type]
                # Shuffle each class's samples for randomness
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
                
        elif split_strategy == 'angle-based':
            # All samples are already filtered by angle in the collection phase
            # but we still need to limit samples per class if requested
            if self.samples_per_class is not None and self.samples_per_class > 0:
                # Group samples by action type
                samples_by_class = {}
                for action_type in self.action_types:
                    samples_by_class[action_type] = [s for s in all_samples if s['action_type'] == action_type]
                    np.random.shuffle(samples_by_class[action_type])
                    # Limit samples per class
                    samples_by_class[action_type] = samples_by_class[action_type][:self.samples_per_class]
                
                # Combine limited samples
                limited_samples = []
                for samples in samples_by_class.values():
                    limited_samples.extend(samples)
                
                self.samples = limited_samples
            else:
                self.samples = all_samples
                
        elif split_strategy == 'random-subset':
            # Group samples by action type
            samples_by_class = {}
            for action_type in self.action_types:
                samples_by_class[action_type] = [s for s in all_samples if s['action_type'] == action_type]
                # Shuffle each class's samples for randomness
                np.random.shuffle(samples_by_class[action_type])
            
            # For training, take the first N samples per class
            # For validation, take the next M samples
            # For test, take the remaining samples
            train_samples = []
            val_samples = []
            test_samples = []
            
            for action_type, samples in samples_by_class.items():
                if self.samples_per_class is not None and self.samples_per_class > 0:
                    max_train = min(self.samples_per_class, len(samples) // 2)  # Ensure we leave some for val/test
                    max_val = min(self.samples_per_class // 2, (len(samples) - max_train) // 2)  # Half as many for val
                    
                    train_samples.extend(samples[:max_train])
                    val_samples.extend(samples[max_train:max_train + max_val])
                    test_samples.extend(samples[max_train + max_val:])
                else:
                    # Default behavior without sample limit
                    n_train = len(samples) // 2
                    n_val = len(samples) // 4
                    
                    train_samples.extend(samples[:n_train])
                    val_samples.extend(samples[n_train:n_train + n_val])
                    test_samples.extend(samples[n_train + n_val:])
            
            # Shuffle the samples within each split
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
        else:
            raise ValueError("split_strategy must be one of 'default', 'angle-based', or 'random-subset'")
        
        # Count samples with noisy data
        noisy_radar_count = sum(1 for s in self.samples if s['noisy_radar_path'] is not None)
        noisy_rfid_count = sum(1 for s in self.samples if s['noisy_rfid_path'] is not None)
        # Count samples from each angle
        angle_counts = {angle: sum(1 for s in self.samples if s['angle'] == angle) for angle in angle_folders}
        # Count samples from each class
        class_counts = {}
        for action_type in self.action_types:
            class_counts[action_type] = sum(1 for s in self.samples if s['action_type'] == action_type)
        # Count samples for each object
        obj_counts = {}
        for i in range(6):  # Assuming 6 object classes (0-5)
            obj_counts[i] = sum(1 for s in self.samples if s['obj_label'] == i)
        
        print(f"Loaded {len(self.samples)} {split} samples from {len(self.action_types)} action types")
        print(f"  - Split strategy: {split_strategy}")
        if split_strategy == 'angle-based':
            print(f"    - Train angles: {train_angles}")
            print(f"    - Val angle: {val_angle}")
        if samples_per_class is not None:
            print(f"    - Max samples per class: {samples_per_class}")
        print(f"  - {noisy_radar_count} samples have noisy radar data available")
        print(f"  - {noisy_rfid_count} samples have noisy RFID data available")
        print(f"  - Samples per angle: {angle_counts}")
        print(f"  - Using {'all angles' if use_multi_angle else 'only 90-degree angle'}")
        print(f"  - Using {'noisy data when available' if use_noisy else 'clean data only'}")
        print(f"  - Samples per action class:")
        for action_type, count in class_counts.items():
            print(f"      {action_type}: {count}")
        print(f"  - Samples per object class:")
        for obj_id, count in obj_counts.items():
            print(f"      Object {obj_id}: {count}")
        print(f"Action types: {self.action_types}")
        print(f"Object names: {self.object_names}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load radar data - CHANGED FOR CLASSIC FORMAT
        sim_radar_data = np.load(sample['sim_radar_path'])  # Now using specs.npy - shape (time_steps, 3, 256, 256)
        
        # Stack spectrograms to make the data shape consistent with the model expectation
        radar_tensor = torch.from_numpy(sim_radar_data).float()
        
        # Load noisy radar data if available and requested
        if self.use_noisy and sample['noisy_radar_path'] is not None:
            noisy_radar_data = np.load(sample['noisy_radar_path'])  # Now using sim2real_specs.npy
            noisy_specs = torch.from_numpy(noisy_radar_data).float()
        else:
            # Fall back to using simulated data
            noisy_specs = radar_tensor
        
        # Load RFID data
        clean_rfid_data = np.load(sample['clean_rfid_path'])
        
        # Load noisy RFID data if available and requested
        if self.use_noisy and sample['noisy_rfid_path'] is not None:
            noisy_rfid_data = np.load(sample['noisy_rfid_path'])
        else:
            # Fall back to using clean data
            noisy_rfid_data = clean_rfid_data
        
        # Load object names data and create one-hot encodings
        obj_names = np.load(sample['obj_names_path'], allow_pickle=True)
        
        # Create one-hot encodings for each object
        obj_one_hot_encodings = []
        for obj_name in obj_names:
            # Create a one-hot vector for this object
            one_hot = np.zeros(len(self.object_names))
            if obj_name in self.object_to_idx:
                one_hot[self.object_to_idx[obj_name]] = 1.0
            obj_one_hot_encodings.append(one_hot)
        
        # Convert to tensor and reshape to match RFID data format
        obj_one_hot_encodings = np.array(obj_one_hot_encodings)
        obj_one_hot_tensor = torch.from_numpy(obj_one_hot_encodings).float()
        
        # Expand the one-hot encodings to match the time dimension of RFID data
        time_steps = clean_rfid_data.shape[0]
        expanded_obj_one_hot = obj_one_hot_tensor.unsqueeze(0).expand(time_steps, -1, -1)  # (time_steps, 6, 8)
        
        # Convert RFID data to PyTorch tensors
        clean_rfid_data = torch.from_numpy(clean_rfid_data).float()
        noisy_rfid_data = torch.from_numpy(noisy_rfid_data).float()
        
        # Concatenate one-hot encodings with RFID data along the last dimension
        # Assuming clean_rfid_data and noisy_rfid_data have shape (time_steps, 6, 4)
        # Add the one-hot encodings to create (time_steps, 6, 4+8)
        clean_rfid_with_obj = torch.cat([clean_rfid_data, expanded_obj_one_hot], dim=-1)
        noisy_rfid_with_obj = torch.cat([noisy_rfid_data, expanded_obj_one_hot], dim=-1)
        
        # Get labels
        action_label = sample['action_label']
        obj_label = sample['obj_label']
        
        # Get angle
        angle = int(sample['angle'])
        
        return {
            'radar_data': radar_tensor,  # (time_steps, 3, 256, 256)
            'noisy_radar_data': noisy_specs_combined,
            'clean_rfid_data': clean_rfid_with_obj,  # RFID data with object one-hot encodings
            'noisy_rfid_data': noisy_rfid_with_obj,
            'action_label': action_label,  # Labels
            'obj_label': obj_label,
            'angle': angle
        }

def new_collate_fn(batch):
    """
    Custom collate function to handle variable length sequences for both radar and RFID data
    """
    # Extract components from batch - Radar data (updated for new format)
    radar_data = [item['radar_data'] for item in batch]  # Each is (time_steps, 3, 256, 256)
    noisy_radar_data = [item['noisy_radar_data'] for item in batch]
    
    # Extract components from batch - RFID data (including one-hot encoding of object names)
    clean_rfid_data = [item['clean_rfid_data'] for item in batch]
    noisy_rfid_data = [item['noisy_rfid_data'] for item in batch]
    
    # Extract components from batch - Labels and angle
    action_labels = [item['action_label'] for item in batch]
    obj_labels = [item['obj_label'] for item in batch]
    angles = [item['angle'] for item in batch]
    
    # Get lengths for padding masks - Radar
    radar_lengths = [data.shape[0] for data in radar_data]
    radar_max_len = max(radar_lengths)
    
    # Get lengths for padding masks - RFID
    rfid_lengths = [data.shape[0] for data in clean_rfid_data]
    rfid_max_len = max(rfid_lengths)
    
    # Create padding masks - Radar
    radar_padding_masks = torch.zeros((len(batch), radar_max_len), dtype=torch.bool)
    for i, length in enumerate(radar_lengths):
        radar_padding_masks[i, length:] = True  # True indicates padding
    
    # Create padding masks - RFID
    rfid_padding_masks = torch.zeros((len(batch), rfid_max_len), dtype=torch.bool)
    for i, length in enumerate(rfid_lengths):
        rfid_padding_masks[i, length:] = True  # True indicates padding
    
    # Pad sequences - Radar
    padded_radar_data = pad_sequence(radar_data, batch_first=True)  # (batch_size, max_time_steps, 3, 256, 256)
    padded_noisy_radar_data = pad_sequence(noisy_radar_data, batch_first=True)
    
    # Pad sequences - RFID
    padded_clean_rfid_data = pad_sequence(clean_rfid_data, batch_first=True)
    padded_noisy_rfid_data = pad_sequence(noisy_rfid_data, batch_first=True)
    
    # Convert labels to tensors
    action_labels = torch.tensor(action_labels)
    obj_labels = torch.tensor(obj_labels)
    angles = torch.tensor(angles)
    
    return {
        # Radar data with mask
        'radar_data': [padded_radar_data, radar_padding_masks],
        'noisy_radar_data': [padded_noisy_radar_data, radar_padding_masks],
        
        # RFID data with mask
        'rfid_data': [padded_clean_rfid_data, rfid_padding_masks],
        'noisy_rfid_data': [padded_noisy_rfid_data, rfid_padding_masks],
        
        # Labels
        'action_labels': action_labels,
        'obj_labels': obj_labels,
        'angles': angles
    }

class RealDatasetwithRFID(Dataset):
    """Dataset class for loading real-world radar-RFID data for human-object interaction recognition.

    Two splitting strategies are supported:
    1. Random-subset split (default):
       From every action class, ``random_subset_n`` samples are drawn uniformly at random (seed=42) to
       constitute the training split.  The remaining samples form the validation split.
    2. Angle-based split:
       If ``val_angle`` (e.g. "90") is provided, all samples whose ``angle`` equals ``val_angle`` are used
       for validation, while the remaining samples are assigned to training.
    """
    def __init__(
        self,
        base_dir: str = '/scratch/tshu2/jyu197/Focus_processed_multi_angle_rfid_real',
        split: str = 'train',
        use_multi_angle: bool = True,
        random_subset_n: int = 37,
        val_angle: Optional[str] = None,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.random_subset_n = random_subset_n
        self.val_angle = val_angle  # if not None we use angle-based splitting

        # Supported splits
        if split not in ['train', 'val']:
            raise ValueError("split must be either 'train' or 'val'")

        # Define the list of possible objects and their mapping to indices (same as synthetic dataset)
        self.object_names = ['bottle', 'pen', 'microwave', 'cabinet', 'phone', 'chair', 'book', 'table']
        self.object_to_idx = {obj: idx for idx, obj in enumerate(self.object_names)}

        # Discover all action types available in the real-world dataset
        self.action_types = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        self.action_types.sort()  # keep deterministic ordering
        self.action_to_label = {action: idx for idx, action in enumerate(self.action_types)}

        # Decide which angles to iterate over
        angle_folders = ['0', '90', '180', '270'] if use_multi_angle else ['90']

        # ------------------------------------------------------------------
        # First pass – load the per-action/angle target_labels.npy files
        # ------------------------------------------------------------------
        self.object_labels: dict[str, np.ndarray | None] = {}
        for action_type in self.action_types:
            for angle in angle_folders:
                angle_path = os.path.join(base_dir, action_type, angle)
                if not os.path.exists(angle_path):
                    print(f"Warning: No '{angle}' folder found for action type {action_type}")
                    self.object_labels[f"{action_type}_{angle}"] = None
                    continue

                target_labels_path = os.path.join(angle_path, 'target_labels.npy')
                if os.path.exists(target_labels_path):
                    try:
                        self.object_labels[f"{action_type}_{angle}"] = np.load(target_labels_path)
                        print(f"Loaded {len(self.object_labels[f'{action_type}_{angle}'])} object labels for {action_type}_{angle}")
                    except Exception as e:
                        print(f"Error loading {target_labels_path}: {e}")
                        self.object_labels[f"{action_type}_{angle}"] = None
                else:
                    print(f"Warning: No target_labels.npy found for {action_type}_{angle}")
                    self.object_labels[f"{action_type}_{angle}"] = None

        # ------------------------------------------------------------------
        # Second pass – enumerate every valid sample folder
        # ------------------------------------------------------------------
        all_samples: list[dict] = []
        for action_type in self.action_types:
            for angle in angle_folders:
                angle_path = os.path.join(base_dir, action_type, angle)
                if not os.path.exists(angle_path):
                    continue

                # Skip if we failed to load object labels for this (action, angle)
                if self.object_labels.get(f"{action_type}_{angle}") is None:
                    print(f"Skipping {action_type}_{angle}: No object labels available")
                    continue

                numbered_folders = [d for d in os.listdir(angle_path) if os.path.isdir(os.path.join(angle_path, d))]
                try:
                    numbered_folders = sorted(numbered_folders, key=int)
                except ValueError:
                    print(f"Warning: Folders in {angle_path} are not numeric – using lexicographic sort")
                    numbered_folders.sort()

                for idx_in_angle, folder in enumerate(numbered_folders):
                    if idx_in_angle >= len(self.object_labels[f"{action_type}_{angle}"]):
                        print(
                            f"Skipping {os.path.join(angle_path, folder)}: Index {idx_in_angle} exceeds available object labels"
                        )
                        continue

                    folder_path = os.path.join(angle_path, folder)

                    # CHANGED FOR CLASSIC FORMAT - looking for specs.npy instead of focus_specs.npz
                    radar_path = os.path.join(folder_path, 'specs.npy') if os.path.exists(os.path.join(folder_path, 'specs.npy')) else None

                    # Real-world RFID phases file
                    rfid_path = os.path.join(folder_path, 'rfid_phases.npy') if os.path.exists(os.path.join(folder_path, 'rfid_phases.npy')) else None

                    # The dataset also contains raw radar cubes (radar_frames.npy). We ignore them here but keep the
                    # path in case future extensions need them.
                    radar_cube_path = os.path.join(folder_path, 'radar_frames.npy') if os.path.exists(os.path.join(folder_path, 'radar_frames.npy')) else None
                    # (The cube path is stored but not loaded to keep memory usage low.)
                    if radar_cube_path:
                        pass  # placeholder for potential future use

                    obj_names_path = os.path.join(folder_path, 'obj_names.npy')

                    if radar_path and rfid_path and os.path.exists(obj_names_path):
                        all_samples.append(
                            {
                                'radar_path': radar_path,
                                'rfid_path': rfid_path,
                                'obj_names_path': obj_names_path,
                                'action_label': self.action_to_label[action_type],
                                'obj_label': int(self.object_labels[f"{action_type}_{angle}"][idx_in_angle]),
                                'action_type': action_type,
                                'angle': angle,
                                'folder_idx': idx_in_angle,
                            }
                        )
                    else:
                        missing = []
                        if not radar_path:
                            missing.append('radar specs.npy')
                        if not rfid_path:
                            missing.append('RFID npy')
                        if not os.path.exists(obj_names_path):
                            missing.append('obj_names.npy')
                        print(f"Skipping {folder_path}: missing {', '.join(missing)}")

        # ------------------------------------------------------------------
        # Construct train/val splits according to the chosen strategy
        # ------------------------------------------------------------------
        np.random.seed(42)  # deterministic shuffling
        if self.val_angle is not None:
            # Angle-based split
            train_samples = [s for s in all_samples if s['angle'] != self.val_angle]
            val_samples = [s for s in all_samples if s['angle'] == self.val_angle]
        else:
            # Random-subset split per action_type
            samples_by_class: dict[str, list] = {}
            for action_type in self.action_types:
                samples_by_class[action_type] = [s for s in all_samples if s['action_type'] == action_type]
                np.random.shuffle(samples_by_class[action_type])

            train_samples, val_samples = [], []
            for action_type, class_samples in samples_by_class.items():
                n_train = min(self.random_subset_n, len(class_samples))
                train_samples.extend(class_samples[:n_train])
                val_samples.extend(class_samples[n_train:])

        # Final assignment based on requested split
        if split == 'train':
            self.samples = train_samples
        else:  # 'val'
            self.samples = val_samples

        # ------------------------------------------------------------------
        # Diagnostic printing
        # ------------------------------------------------------------------
        angle_counts = {angle: sum(1 for s in self.samples if s['angle'] == angle) for angle in angle_folders}
        class_counts = {action: sum(1 for s in self.samples if s['action_type'] == action) for action in self.action_types}
        obj_counts = {i: sum(1 for s in self.samples if s['obj_label'] == i) for i in range(len(self.object_names))}

        print(
            f"Loaded {len(self.samples)} {split} samples from {len(self.action_types)} action types (real-world dataset)"
        )
        print(f"  - Split strategy: {'angle='+self.val_angle if self.val_angle else f'random_subset_n={self.random_subset_n}'}")
        print(f"  - Samples per angle: {angle_counts}")
        print(f"  - Samples per action class:")
        for action, cnt in class_counts.items():
            print(f"      {action}: {cnt}")
        print(f"  - Samples per object class:")
        for obj_id, cnt in obj_counts.items():
            print(f"      Object {obj_id}: {cnt}")
        print(f"Action types: {self.action_types}")
        print(f"Object names: {self.object_names}")

    # ----------------------------------------------------------------------
    # PyTorch Dataset protocol
    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ---------------------------- Radar ----------------------------
        # Load and process radar data - CHANGED FOR CLASSIC FORMAT
        radar_data = np.load(sample['radar_path'])  # shape (time_steps, 3, 256, 256)
        
        # Stack spectrograms to make the data shape consistent with the model expectation
        radar_tensor = torch.from_numpy(radar_data).float()
        
        # For compatibility with the training code, we expose the same data for both clean and noisy
        noisy_radar_tensor = radar_tensor

        # ----------------------------- RFID ----------------------------
        rfid_data_np = np.load(sample['rfid_path'])
        clean_rfid_tensor = torch.from_numpy(rfid_data_np).float()
        noisy_rfid_tensor = clean_rfid_tensor  # identical for real-world dataset

        # Object one-hot encoding
        obj_names = np.load(sample['obj_names_path'], allow_pickle=True)
        one_hots = np.zeros((len(obj_names), len(self.object_names)), dtype=np.float32)
        for i, obj_name in enumerate(obj_names):
            if obj_name in self.object_to_idx:
                one_hots[i, self.object_to_idx[obj_name]] = 1.0
        one_hots = torch.from_numpy(one_hots)  # (N_tags, 8)

        time_steps = clean_rfid_tensor.shape[0]
        expanded_one_hots = one_hots.unsqueeze(0).expand(time_steps, -1, -1)  # (T, N_tags, 8)

        clean_rfid_with_obj = torch.cat([clean_rfid_tensor, expanded_one_hots], dim=-1)
        noisy_rfid_with_obj = clean_rfid_with_obj  # identical

        return {
            'radar_data': radar_tensor,  # (time_steps, 3, 256, 256)
            'noisy_radar_data': noisy_radar_tensor,
            'clean_rfid_data': clean_rfid_with_obj,
            'noisy_rfid_data': noisy_rfid_with_obj,
            'action_label': sample['action_label'],
            'obj_label': sample['obj_label'],
            'angle': int(sample['angle']),
        }