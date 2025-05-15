import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import model_utils

class synDataset:
    """
    Load total dataset
    """
    def __init__(self, data_dir, tot_action, use_action, use_angles, scale=1.0):
        """
        Store the filenames of the data to use.

        Args:
            root_dir: (string) directory containing the dataset
            folders: (list) containing the id of the motion segment
        """
        self.data_dir = data_dir
        self.tot_action = tot_action
        self.use_action = use_action
        self.use_angles = use_angles
        self.scale = scale
        # may be data is too large to load at once, then load data on the fly
        self.specs = []
        self.action_label = []
        self.objs = []
        self.obj_names = []  # the embedding of object names
        self.obj_label = []
        self.load_data()
        unique,count=np.unique(self.action_label,return_counts=True)
        data_count=dict(zip(unique,count))
        print("action label count: ", data_count)    
        unique,count=np.unique(self.obj_label,return_counts=True)
        data_count=dict(zip(unique,count))
        print("obj label count: ", data_count)    
        
    def load_data(self):
        for i in range(len(self.tot_action)):
           action = self.tot_action[i]
           if action not in self.use_action:
                continue
           for angle in self.use_angles:
                action_angle_path = os.path.join(self.data_dir, action.replace(' ','_'), str(angle))
                if not os.path.exists(action_angle_path):
                    print(f"Warning: Path does not exist {action_angle_path}")
                    continue

                files = [f for f in os.listdir(action_angle_path) if os.path.isdir(os.path.join(action_angle_path, f))]
                obj_file_path = os.path.join(action_angle_path, 'target_labels.npy')
                if not os.path.exists(obj_file_path):
                    print(f"Warning: target_labels.npy not found in {action_angle_path}")
                    continue
                obj_file = np.load(obj_file_path)
                
                length = int(len(files)*self.scale)
                # Ensure files are sorted for consistent selection if length < len(files)
                files.sort() # Sort by name, or use key=int if they are purely numeric strings
                
                for idx_str in files[:length]: # Iterate over selected folder names
                    # Attempt to convert folder name to int if it's purely numeric for indexing obj_file
                    try:
                        numerical_idx = int(idx_str)
                    except ValueError:
                        print(f"Warning: Folder name {idx_str} in {action_angle_path} is not purely numeric. Skipping or implement alternative indexing for obj_file.")
                        continue # Or handle differently if folder names are not guaranteed to be 0, 1, 2...
                    
                    current_data_path = os.path.join(action_angle_path, idx_str)
                    specs_path = os.path.join(current_data_path, 'sim2real_specs.npy')
                    obj_path = os.path.join(current_data_path, 'obj_refined_phase.npy')
                    obj_name_path = os.path.join(current_data_path, 'obj_name_embeddings.npy')
                    real_name_path = os.path.join(current_data_path, 'obj_names.npy')

                    if not all(os.path.exists(p) for p in [specs_path, obj_path, obj_name_path, real_name_path]):
                        print(f"Warning: One or more data files missing in {current_data_path}")
                        continue
                    
                    spec = np.load(specs_path)
                    self.specs.append(spec)
                    self.action_label.append(i) # Using original action index i
                    
                    obj = np.load(obj_path)
                    self.objs.append(obj)
                    if numerical_idx < len(obj_file):
                         self.obj_label.append(obj_file[numerical_idx])
                    else:
                        print(f"Warning: Index {numerical_idx} out of bounds for obj_file in {action_angle_path}. Length is {len(obj_file)}.")
                        # Decide how to handle: skip, append a default, or raise error
                        continue

                    obj_name_data = np.load(obj_name_path)
                    self.obj_names.append(obj_name_data)
                    self.real_name = np.load(real_name_path)
                       
        self.specs = np.array(self.specs, dtype=object) if self.specs else np.empty((0), dtype=object)
        self.action_label = np.array(self.action_label) if self.action_label else np.empty((0), dtype=int)
        self.objs = np.array(self.objs, dtype=object) if self.objs else np.empty((0), dtype=object)
        self.obj_names = np.array(self.obj_names, dtype=object) if self.obj_names else np.empty((0), dtype=object)
        self.obj_label = np.array(self.obj_label) if self.obj_label else np.empty((0), dtype=int)

class realDataset:
    """
    Load total dataset
    """
    def __init__(self, data_dir, tot_action, use_action, use_angles, scale=1.0):
        """
        Store the filenames of the data to use.

        Args:
            root_dir: (string) directory containing the dataset
            folders: (list) containing the id of the motion segment
        """
        self.data_dir = data_dir
        self.tot_action = tot_action
        self.use_action = use_action
        self.use_angles = use_angles
        self.scale = scale
        self.specs = []
        self.action_label = []
        self.objs = []
        self.obj_names = []  # the embedding of object names
        self.obj_label = []
        self.load_data()
        if self.action_label:
            unique,count=np.unique(self.action_label,return_counts=True)
            data_count=dict(zip(unique,count))
            print("action label count: ", data_count)
        if self.obj_label:
            unique,count=np.unique(self.obj_label,return_counts=True)
            data_count=dict(zip(unique,count))
            print("obj label count: ", data_count)
        
    def load_data(self):
        action_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        for action_name_in_dir in action_dirs:
            current_action_str = action_name_in_dir.replace('_',' ')
            if current_action_str not in self.use_action:
                continue
            
            action_path_base = os.path.join(self.data_dir, action_name_in_dir)
            angle_dirs = [d for d in os.listdir(action_path_base) if os.path.isdir(os.path.join(action_path_base, d))]

            for angle_str in angle_dirs:
                try:
                    angle = int(angle_str)
                    if angle not in self.use_angles:
                        continue
                except ValueError:
                    print(f"Warning: Non-integer angle folder '{angle_str}' found in {action_path_base}. Skipping.")
                    continue
                
                current_angle_path = os.path.join(action_path_base, angle_str)
                obj_file_path = os.path.join(current_angle_path, 'target_labels.npy')
                if not os.path.exists(obj_file_path):
                    print(f"Warning: target_labels.npy not found in {current_angle_path}")
                    continue
                obj_file = np.load(obj_file_path)
                
                instance_folders = [d for d in os.listdir(current_angle_path) if os.path.isdir(os.path.join(current_angle_path, d))]
                # Sort instance folders to ensure consistent processing order, especially if scaling samples
                instance_folders.sort() # Assuming numeric or sortable names like '0', '1', '10'
                
                num_samples_to_load = int(len(instance_folders) * self.scale)
                
                for folder_name in instance_folders[:num_samples_to_load]:
                    try:
                        numerical_idx = int(folder_name) # For indexing into obj_file
                    except ValueError:
                        print(f"Warning: Instance folder name {folder_name} in {current_angle_path} is not purely numeric. Skipping or implement alternative indexing.")
                        continue

                    instance_path = os.path.join(current_angle_path, folder_name)
                    specs_path = os.path.join(instance_path, 'specs.npy')
                    obj_path = os.path.join(instance_path, 'rfid_phases.npy')
                    obj_name_path = os.path.join(instance_path, 'obj_name_embeddings.npy')

                    if not all(os.path.exists(p) for p in [specs_path, obj_path, obj_name_path]):
                        print(f"Warning: One or more data files missing in {instance_path}")
                        continue
                    
                    spec = np.load(specs_path)
                    self.specs.append(spec)
                    self.action_label.append(self.tot_action.index(current_action_str))
                    
                    obj = np.load(obj_path)
                    self.objs.append(obj)
                    if numerical_idx < len(obj_file):
                        self.obj_label.append(obj_file[numerical_idx])
                    else:
                        print(f"Warning: Index {numerical_idx} out of bounds for obj_file in {current_angle_path}. Length is {len(obj_file)}.")
                        continue # Or handle missing label appropriately

                    obj_name_data = np.load(obj_name_path)
                    self.obj_names.append(obj_name_data)
                    
        self.specs = np.array(self.specs, dtype=object) if self.specs else np.empty((0), dtype=object)
        self.action_label = np.array(self.action_label) if self.action_label else np.empty((0), dtype=int)
        self.objs = np.array(self.objs, dtype=object) if self.objs else np.empty((0), dtype=object)
        self.obj_names = np.array(self.obj_names, dtype=object) if self.obj_names else np.empty((0), dtype=object)
        self.obj_label = np.array(self.obj_label) if self.obj_label else np.empty((0), dtype=int)

class enhancedDataset:
    """
    Merge the real and synthetic dataset
    """
    def __init__(self, real_dataset, syn_dataset):
        """
        Store the filenames of the data to use.

        Args:
            root_dir: (string) directory containing the dataset
            folders: (list) containing the id of the motion segment
        """
        self.real_dataset = real_dataset
        self.syn_dataset = syn_dataset
        self.specs = np.concatenate((real_dataset.specs, syn_dataset.specs), axis=0)
        self.action_label = np.concatenate((real_dataset.action_label, syn_dataset.action_label), axis=0)
        self.objs = np.concatenate((real_dataset.objs, syn_dataset.objs), axis=0)
        self.obj_names = np.concatenate((real_dataset.obj_names, syn_dataset.obj_names), axis=0)
        self.obj_label = np.concatenate((real_dataset.obj_label, syn_dataset.obj_label), axis=0)
        
        unique,count=np.unique(self.action_label,return_counts=True)
        data_count=dict(zip(unique,count))
        print("action label count: ", data_count)    
        unique,count=np.unique(self.obj_label,return_counts=True)
        data_count=dict(zip(unique,count))
        print("obj label count: ", data_count)    


class mmWaveDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    Compatible with the training code in train_lw_rfid.py.
    """
    def __init__(self, data_dir, split='train', use_multi_angle=True, use_real_data=False, real_data_dir=None,
                 # Split strategy parameters
                 split_strategy='default',  # 'default', 'angle-based', 'random-subset'
                 train_angles=None,  # for angle-based, list of angles to use for training
                 val_angle=None,     # for angle-based, angle to use for validation
                 samples_per_class=None,  # for random-subset, limit samples per class
                 seed=42):
        """
        Initialize the dataset with options compatible with train_lw_rfid.py
        """
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.use_multi_angle = use_multi_angle
        self.use_real_data = use_real_data
        self.real_data_dir = real_data_dir if real_data_dir else data_dir
        self.split_strategy = split_strategy
        self.train_angles = train_angles if train_angles else [0, 180, 270]
        self.val_angle = val_angle if val_angle else 90
        self.samples_per_class = samples_per_class
        self.seed = seed
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Load dataset based on real or simulated setting
        if use_real_data:
            params = self._get_params_for_real_dataset()
            total_dataset = realDataset(self.real_data_dir, params.use_action, params.train_action, params.train_angle, params.datasize_scale)
        else:
            params = self._get_params_for_syn_dataset()
            total_dataset = synDataset(self.data_dir, params.use_action, params.train_action, params.train_angle, params.datasize_scale)

        # Apply split strategy
        self._apply_split_strategy(total_dataset)

    def _get_params_for_syn_dataset(self):
        """Create parameter object for synthetic dataset"""
        class Params:
            pass
        
        params = Params()
        params.use_action = ["close", "open", "pick up", "put down", "sit down", "stand up", "wipe"]
        params.train_action = params.use_action
        
        # Set angles based on split_strategy
        if self.split_strategy == 'angle-based':
            if self.split == 'train':
                params.train_angle = [int(a) for a in self.train_angles]
            else:  # val or test
                params.train_angle = [self.val_angle]
        else:
            # For default or random-subset, we'll filter later
            if self.use_multi_angle:
                params.train_angle = [0, 90, 180, 270]
            else:
                params.train_angle = [90]
                
        params.datasize_scale = 1.0
        return params
        
    def _get_params_for_real_dataset(self):
        """Create parameter object for real dataset"""
        class Params:
            pass
        
        params = Params()
        params.use_action = ["close", "open", "pick up", "put down", "sit down", "stand up", "wipe"]
        params.train_action = params.use_action
        
        # Set angles based on split_strategy
        if self.split_strategy == 'angle-based':
            if self.split == 'train':
                params.train_angle = [int(a) for a in self.train_angles]
            else:  # val or test
                params.train_angle = [self.val_angle]
        else:
            # For default or random-subset, we'll filter later
            if self.use_multi_angle:
                params.train_angle = [0, 90, 180, 270]
            else:
                params.train_angle = [90]
                
        params.datasize_scale = 1.0
        return params
    
    def _apply_split_strategy(self, total_dataset):
        """Apply the requested split strategy to the loaded dataset"""
        # Extract data from the total dataset
        self.specs = total_dataset.specs
        self.action_label = total_dataset.action_label
        self.objs = total_dataset.objs
        self.obj_label = total_dataset.obj_label
        self.obj_names = total_dataset.obj_names
        
        # Create indices for all samples
        all_indices = np.arange(len(self.action_label))
        
        if self.split_strategy == 'default':
            # Split data using default 70/15/15 ratio
            train_indices, test_indices = train_test_split(all_indices, test_size=0.3, random_state=self.seed, stratify=self.action_label)
            val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=self.seed, 
                                                        stratify=self.action_label[test_indices])
            
            # Select the appropriate split
            if self.split == 'train':
                selected_indices = train_indices
            elif self.split == 'val':
                selected_indices = val_indices
            else:  # test
                selected_indices = test_indices
                
        elif self.split_strategy == 'angle-based':
            # Filtering is already done during the data loading in _get_params_for_*_dataset
            selected_indices = all_indices
            
        elif self.split_strategy == 'random-subset':
            # Group samples by action type
            action_types = np.unique(self.action_label)
            samples_by_class = {}
            for action_type in action_types:
                samples_by_class[action_type] = all_indices[self.action_label == action_type]
                # Shuffle each class's samples for randomness
                np.random.shuffle(samples_by_class[action_type])
            
            # For random-subset, only split into train and val
            train_indices = []
            val_indices = []
            
            for action_type, indices in samples_by_class.items():
                if self.samples_per_class is not None and self.samples_per_class > 0:
                    # Take up to samples_per_class for training
                    train_count = min(self.samples_per_class, len(indices))
                    train_indices.extend(indices[:train_count])
                    
                    # Remaining samples go to validation
                    val_indices.extend(indices[train_count:])
                else:
                    # Default: 70% train, 30% val if no specific sample count provided
                    train_idx = int(len(indices) * 0.7)
                    
                    train_indices.extend(indices[:train_idx])
                    val_indices.extend(indices[train_idx:])
            
            # Shuffle the indices within each split
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)
            
            # Select the appropriate split
            if self.split == 'train':
                selected_indices = train_indices
            else:  # val or test
                selected_indices = val_indices
                
        # Filter the data based on selected indices
        self.specs = self.specs[selected_indices]
        self.action_label = self.action_label[selected_indices]
        self.objs = self.objs[selected_indices]
        self.obj_label = self.obj_label[selected_indices]
        self.obj_names = self.obj_names[selected_indices]
        
        # Print dataset statistics
        unique, count = np.unique(self.action_label, return_counts=True)
        print(f"Split: {self.split}, Strategy: {self.split_strategy}")
        print(f"Action label counts: {dict(zip(unique, count))}")
        
    def __len__(self):
        # return size of dataset
        return self.action_label.shape[0]

    def __getitem__(self, idx):
        """
        Fetch index idx signal, objs, and labels from dataset in a format compatible with train_lw_rfid.py.

        Returns:
            Dictionary with:
            - sim_specs: tensor of shape (time_steps, 3, 256, 256) - simulated spectrogram data
            - noisy_specs: tensor of shape (time_steps, 3, 256, 256) - noisy/real spectrogram data
            - label: action label tensor
            - angle: angle information (dummy value for compatibility)
        """
        # Get mmWave data - will be used with model's mmWaveBranch
        spec = torch.from_numpy(self.specs[idx].astype(np.float32))
        
        # Get object data - will be used with model's OBJBranch
        obj = torch.from_numpy(self.objs[idx].astype(np.float32))
        obj_name = torch.from_numpy(np.array(self.obj_names[idx]))
        
        # Get labels
        action_label = int(self.action_label[idx])
        obj_label = int(self.obj_label[idx])
        
        # Use the same data for sim_specs and noisy_specs in this implementation
        # The training code will select which one to use based on args.use_noisy
        return {
            'sim_specs': spec,
            'noisy_specs': spec,  # Use the same for simplicity
            'label': action_label,
            'angle': 0,  # Dummy value for compatibility
            'obj': obj,  # Added for HOINet compatibility
            'obj_label': obj_label,  # Added for HOINet compatibility
            'obj_name': obj_name,  # Added for HOINet compatibility
            'idx': idx  # Added for tracking
        }

def lw_collate_fn(batch):
    """
    Custom collate function compatible with train_lw_rfid.py and HOINet model.
    Similar to the lw_collate_fn from the original code but modified to include RFID data.
    """
    # Extract mmWave components from batch
    sim_specs = [item['sim_specs'] for item in batch]
    noisy_specs = [item['noisy_specs'] for item in batch]
    labels = [item['label'] for item in batch]
    angles = [item['angle'] for item in batch]
    
    # Extract RFID components from batch
    objs = [item['obj'] for item in batch]
    obj_labels = [item['obj_label'] for item in batch]
    obj_names = [item['obj_name'] for item in batch]
    
    # Get lengths for padding masks
    mm_lengths = [spec.shape[0] for spec in sim_specs]
    obj_lengths = [obj.shape[0] for obj in objs]
    max_mm_len = max(mm_lengths)
    max_obj_len = max(obj_lengths)
    
    # Create padding masks
    mm_padding_masks = torch.zeros((len(batch), max_mm_len), dtype=torch.bool)
    obj_padding_masks = torch.zeros((len(batch), max_obj_len), dtype=torch.bool)
    
    for i, length in enumerate(mm_lengths):
        mm_padding_masks[i, length:] = True  # True indicates padding
    
    for i, length in enumerate(obj_lengths):
        obj_padding_masks[i, length:] = True  # True indicates padding
    
    # Pad sequences
    padded_sim_specs = pad_sequence(sim_specs, batch_first=True)
    padded_noisy_specs = pad_sequence(noisy_specs, batch_first=True)
    padded_objs = pad_sequence(objs, batch_first=True)
    
    # Convert labels to tensors
    labels = torch.tensor(labels)
    angles = torch.tensor(angles)
    obj_labels = torch.tensor(obj_labels)
    
    # Stack obj_names
    obj_names = torch.stack(obj_names)
    
    # Return in a format compatible with both train_lw_rfid.py and HOINet model
    return {
        'sim_specs': padded_sim_specs,
        'noisy_specs': padded_noisy_specs,
        'padding_mask': mm_padding_masks,
        'labels': labels,
        'angles': angles,
        'obj': padded_objs,
        'obj_mask': obj_padding_masks,
        'obj_name': obj_names,
        'obj_labels': obj_labels
    }

def fetch_dataloader(types, real_dir, syn_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        real_dir: (string) directory containing real dataset
        syn_dir: (string) directory containing simulated dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    # Convert training arguments from params to the format needed by mmWaveDataset
    use_multi_angle = params.use_multi_angle if hasattr(params, 'use_multi_angle') else True
    split_strategy = params.split_strategy if hasattr(params, 'split_strategy') else 'default'
    
    # Determine which angles to use based on dataset type
    if hasattr(params, 'train_angles'):
        train_angles = params.train_angles
    else:
        train_angles = [0, 180, 270]
        
    if hasattr(params, 'val_angle'):
        val_angle = params.val_angle
    else:
        val_angle = 90
        
    # Determine samples per class for random-subset strategy
    samples_per_class = params.samples_per_class if hasattr(params, 'samples_per_class') else None
    
    # Determine dataset type (real, simulated, or mixed)
    dataset_type = params.dataset_type if hasattr(params, 'dataset_type') else 'simulated'
    
    for split in ['train', 'val']:
        if split in types:
            if dataset_type == 'mixed':
                # Create both simulated and real datasets
                sim_dataset = mmWaveDataset(
                    data_dir=syn_dir, 
                    split=split,
                    use_multi_angle=use_multi_angle,
                    use_real_data=False,
                    split_strategy=split_strategy,
                    train_angles=train_angles,
                    val_angle=val_angle,
                    samples_per_class=samples_per_class
                )
                
                real_dataset = mmWaveDataset(
                    data_dir=real_dir,
                    split=split,
                    use_multi_angle=use_multi_angle,
                    use_real_data=True,
                    real_data_dir=real_dir,
                    split_strategy=split_strategy,
                    train_angles=train_angles,
                    val_angle=val_angle,
                    samples_per_class=samples_per_class
                )
                
                # Use MixedDataLoader from train_lw_rfid.py
                # This is just a placeholder - you'll need to implement MixedDataLoader separately
                dl = MixedDataLoader(
                    sim_dataset=sim_dataset,
                    real_dataset=real_dataset,
                    batch_size=params.batch_size,
                    shuffle=(split == 'train'),
                    num_workers=params.num_workers,
                    collate_fn=lw_collate_fn,
                    pin_memory=params.cuda
                )
                
            else:
                # Create either real or simulated dataset based on dataset_type
                dataset = mmWaveDataset(
                    data_dir=real_dir if dataset_type == 'real' else syn_dir,
                    split=split,
                    use_multi_angle=use_multi_angle,
                    use_real_data=(dataset_type == 'real'),
                    real_data_dir=real_dir,
                    split_strategy=split_strategy,
                    train_angles=train_angles,
                    val_angle=val_angle,
                    samples_per_class=samples_per_class
                )
                
                dl = DataLoader(
                    dataset, 
                    batch_size=params.batch_size, 
                    shuffle=(split == 'train'),
                    num_workers=params.num_workers,
                    collate_fn=lw_collate_fn,
                    pin_memory=params.cuda,
                    drop_last=(split == 'train')
                )
            
            dataloaders[split] = dl

    return dataloaders

# Simple MixedDataLoader implementation - you may need to customize this
class MixedDataLoader:
    """
    A simplified version of the MixedDataLoader from train_lw_rfid.py.
    Interleaves batches from simulated and real datasets.
    """
    def __init__(self, sim_dataset, real_dataset, batch_size, shuffle=True, 
                 num_workers=1, collate_fn=None, drop_last=False, pin_memory=True):
        # Build the underlying DataLoaders
        self.sim_loader = DataLoader(
            sim_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        self.real_loader = DataLoader(
            real_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        self.shuffle = shuffle

    def __iter__(self):
        # Fresh iterators for each epoch
        self.sim_iter = iter(self.sim_loader)
        self.real_iter = iter(self.real_loader)

        # Build the epoch sequence of sources ("sim" / "real")
        self.batch_sequence = ["sim"] * len(self.sim_loader) + [
            "real"
        ] * len(self.real_loader)
        
        if self.shuffle:
            random.shuffle(self.batch_sequence)
        
        self._seq_idx = 0
        return self

    def __next__(self):
        while self._seq_idx < len(self.batch_sequence):
            src = self.batch_sequence[self._seq_idx]
            self._seq_idx += 1

            if src == "sim":
                try:
                    batch = next(self.sim_iter)
                    return batch, src
                except StopIteration:
                    continue
            else:  # src == "real"
                try:
                    batch = next(self.real_iter)
                    return batch, src
                except StopIteration:
                    continue
        
        # Both iterators are exhausted
        raise StopIteration

    def __len__(self):
        return len(self.sim_loader) + len(self.real_loader)


if __name__ == '__main__':
    syn_dir  = '/weka/scratch/rzhao36/lwang/datasets/HOI/RealAction/datasets/classic_syn/'
    real_dir  = '/weka/scratch/rzhao36/lwang/datasets/HOI/RealAction/datasets/classic/'
    setting_dir = './data/exp_settings/seen_unseen_action/'   # Directory containing params.json

    json_path = os.path.join(setting_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = model_utils.Params(json_path)
    dataloaders = fetch_dataloader(['train', 'val'], real_dir, syn_dir, params)
        