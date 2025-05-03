import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import model_utils

class totalDataset:
    """
    Load total dataset
    """
    def __init__(self, data_dir, use_action,use_angles,scale=1.0):
        """
        Store the filenames of the data to use.

        Args:
            root_dir: (string) directory containing the dataset
            folders: (list) containing the id of the motion segment
        """
        self.data_dir = data_dir
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
        for i in range(len(self.use_action)):
           action = self.use_action[i]
           for angle in self.use_angles:
                load_path = self.data_dir+action.replace(' ','_')+'/'+str(angle)+'/'
                files = [f for f in os.listdir(load_path) if os.path.isdir(os.path.join(load_path, f))]
                obj_file = np.load(load_path+'target_labels.npy')
                length = int(len(files)*self.scale)
                for idx in range(length):
                    #specs_path = load_path +str(idx) + '/sim2real_specs.npy'
                    specs_path = load_path +str(idx) + '/specs.npy'
                    spec = np.load(specs_path)
                    self.specs.append(spec)
                    self.action_label.append(i)
                    
                    #obj_path = load_path +str(idx) + '/obj_refined_phase.npy'
                    obj_path = load_path +str(idx) + '/obj_clean_phase.npy'
                    obj = np.load(obj_path)
                    self.objs.append(obj)
                    self.obj_label.append(obj_file[idx])
                    obj_name_path = load_path +str(idx) + '/obj_name_embeddings.npy'
                    obj_name = np.load(obj_name_path)
                    self.obj_names.append(obj_name)
                    self.real_name = np.load(load_path +str(idx) + '/obj_names.npy')
                       
               
        self.specs = np.array(self.specs, dtype=object)
        self.action_label = np.array(self.action_label)
        self.objs = np.array(self.objs, dtype=object)
        self.obj_names = np.array(self.obj_names)
        self.obj_label = np.array(self.obj_label)
        #for i in range(self.obj_label.shape[0]):
        #    target_id = self.obj_label[i]
        #    print("obj names",target_id,self.real_name[target_id],self.real_name)
     

class mmWaveDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, total_dataset, data_idx):
        """
        Store the filenames of the data to use.

        Args:
            root_dir: (string) directory containing the dataset
            folders: (list) containing the id of the motion segment
        """


        # load labels
        self.action_label = total_dataset.action_label[data_idx]
        # may be data is too large to load at once, then load data on the fly
        self.specs = total_dataset.specs[data_idx]  
        self.objs = total_dataset.objs[data_idx]
        self.obj_label = total_dataset.obj_label[data_idx]
        self.obj_names = total_dataset.obj_names[data_idx]
        
    def __len__(self):
        # return size of dataset
        return self.action_label.shape[0]

    def __getitem__(self, idx):
        """
        Fetch index idx signal, objs, and labels from dataset. 

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            spec: (tensor) three radar images for each frame 
            action_label: action label.
            obj: object location
            obj_label: object label
            obj_name: object name embedding
        """
        spec = torch.from_numpy(self.specs[idx].astype(np.float32))
        action_label = torch.from_numpy(np.array(self.action_label[idx])).unsqueeze(-1)
        obj = torch.from_numpy(self.objs[idx].astype(np.float32))
        obj_label = torch.from_numpy(np.array(self.obj_label[idx])).unsqueeze(-1)
        obj_name = torch.from_numpy(np.array(self.obj_names[idx]))
        return spec, action_label,obj,obj_label, obj_name, idx
        
def collate_fn(batch):
    specs, action_labels, objs,obj_labels,obj_name ,idxs = zip(*batch)
    
    def pad_mask(data):     
        # get origin length
        lengths = [x.shape[0] for x in data] 
        # zero-padding
        padded_data = pad_sequence(data, batch_first=True, padding_value=0) 
        # mask for zero-padding
        max_seq_len = padded_data.shape[1]
        mask = torch.zeros((len(data), max_seq_len), dtype=torch.bool) 
        for i, length in enumerate(lengths):
            mask[i, length:] = True
        return padded_data, mask
    
    padded_specs, specs_mask = pad_mask(specs)
    padded_obj, obj_mask = pad_mask(objs)
    action_labels = torch.tensor(action_labels)
    obj_labels = torch.tensor(obj_labels)
    idxs = torch.tensor(idxs)
    obj_name = torch.stack(obj_name)
    return [padded_specs, specs_mask,action_labels], [padded_obj,obj_mask,obj_name,obj_labels],idxs

def fetch_dataloader(types, root_dir,params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    use_angle = params.use_angle
    use_action = params.use_action
    scale = params.datasize_scale
    data_split_ratio = params.data_split_ratio
    total_dataset = totalDataset(root_dir, use_action,use_angle,scale)

    idx = np.arange(total_dataset.action_label.shape[0])
    train_idx, test_idx = train_test_split(idx, train_size=data_split_ratio[0], random_state=2025)
    val_idx, test_idx = train_test_split(test_idx, train_size=data_split_ratio[1], random_state=2025)
    print("actions:",use_action,"angle: ", use_angle, "train size: ", len(train_idx), 
          "val size: ", len(val_idx), "test size: ", len(test_idx))
    print("train size: ", len(train_idx), "val size: ", len(val_idx), "test size: ", len(test_idx))
    for split in ['train', 'val', 'test']:
        if split in types:
            
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dataset = mmWaveDataset(total_dataset,train_idx)
                dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=True,    
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda,
                                        collate_fn=collate_fn)
            elif split == 'val':
                dataset = mmWaveDataset(total_dataset,val_idx)
                dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda,
                                        collate_fn=collate_fn)
            else:
                dataset = mmWaveDataset(total_dataset,test_idx)
                dl = DataLoader(dataset, batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda,
                                        collate_fn=collate_fn)
            dataloaders[split] = dl

    return dataloaders

if __name__ == '__main__':
    root_dir  = "/weka/scratch/rzhao36/lwang/datasets/HOI/datasets/classic/"
    setting_dir = './data/exp_settings/hoi/'   # Directory containing params.json

    json_path = os.path.join(setting_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = model_utils.Params(json_path)
    dataloaders = fetch_dataloader(['train', 'val', 'test'], root_dir, params)
    #for split in ['train', 'val', 'test']:
    #    print(split)
    #    preds = []
    #    GTs = []
    #    for i, (mm_data_list, rfid_data_list, idxs) in enumerate(dataloaders[split]):
    #        padded_syn_specs, syn_specs_mask,syn_action_labels = mm_data_list
    #        padded_obj,obj_mask,obj_name,obj_labels = rfid_data_list
            
    #        padded_obj = padded_obj.cpu().numpy()
    #        obj_mask = obj_mask.cpu().numpy()
    #        obj_labels = obj_labels.cpu().numpy()
    #        objs = []
    #        for i in range(padded_obj.shape[0]):  # 遍历每个样本
    #            valid_indices = np.where(obj_mask[i] == False)[0]  # 找到有效时间步的索引
    #            objs.append(padded_obj[i, valid_indices])  # 提取有效时间步的数据
    #        for i in range(len(objs)):
    #            obj = objs[i]
    #            y_differences = obj[:,:,1]
    #            metric = np.sum(y_differences,axis=0)
    #            abs_metric = np.abs(metric)
    #            pred = np.argmax(abs_metric)
    #            preds.append(pred)
    #            GTs.append(obj_labels[i])
    #    preds = np.array(preds)
    #    GTs = np.array(GTs)
    #    accuracy = np.sum(preds==GTs)/preds.shape[0]
    #    print('Accuracy:',accuracy)
        