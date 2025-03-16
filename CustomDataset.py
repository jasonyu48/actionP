import logging
import torch
import os
from torch.utils.data.dataset import Dataset
import numpy as np
from glob import glob

log = logging.getLogger(__name__)

class XRFProcessedDataset(Dataset):
    def __init__(self, base_dir='./xrf555_processed', is_train=True, train_ratio=0.9):
        super().__init__()
        self.base_dir = base_dir
        self.is_train = is_train
        self.train_ratio = train_ratio
        
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
        
        # Split into train and test
        split_idx = int(len(all_samples) * train_ratio)
        self.samples = all_samples[:split_idx] if is_train else all_samples[split_idx:]
        
        dataset_type = "training" if is_train else "test"
        print(f"Loaded {len(self.samples)} {dataset_type} samples from {len(self.action_types)} action types")
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


# Keep the original dataset class for backward compatibility
class XRFBertDatasetNewMix(Dataset):
    def __init__(self, file_path='./dataset/XRFDataset/', is_train=True, scene='dml'):
        super(XRFBertDatasetNewMix, self).__init__()
        self.word_list = np.load("./word2vec/bert_new_sentence_large_uncased.npy")
        self.file_path = file_path
        self.is_train = is_train
        self.scene = scene
        if self.is_train:
            self.file = self.file_path + self.scene + '_train.txt'
        else:
            self.file = self.file_path + self.scene + '_val.txt'
        file = open(self.file)
        val_list = file.readlines()
        self.data = {
            'file_name': list(),
            'label': list()
        }
        self.path = self.file_path + self.scene + '_new_data/'
        for string in val_list:
            self.data['file_name'].append(string.split(',')[0])
            self.data['label'].append(int(string.split(',')[2]) - 1)
        log.info("load XRF dataset")

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        file_name = self.data['file_name'][idx]
        label = self.data['label'][idx]
        vector = self.word_list[label]

        wifi_data = load_wifi(file_name, self.is_train, path=self.path)
        rfid_data = load_rfid(file_name, self.is_train, path=self.path)
        mmwave_data = load_mmwave(file_name, self.is_train, path=self.path)
        return wifi_data, rfid_data, mmwave_data, label, vector


def load_rfid(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    record = np.load(path + 'RFID/' + filename + ".npy")
    return torch.from_numpy(record).float()


def load_wifi(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    record = np.load(path + 'WiFi/' + filename + ".npy")
    return torch.from_numpy(record).float()


def load_mmwave(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    mmWave_data = np.load(path + 'mmWave/' + filename + ".npy")
    return torch.from_numpy(mmWave_data).float()

