import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import model_utils
import smplx
import math
class Reshape(torch.nn.Module):
    def __init__(self,type):
        super(Reshape, self).__init__()
        self._type = type

    def forward(self, x):
        if self._type == -1:
            return x.permute(0, 2, 1, 3)
        else:
            return x.view(-1, self._type)
class Squeeze(torch.nn.Module):
    def __init__(self,type):
        super(Squeeze, self).__init__()
        self._type = type

    def forward(self, x):
        if self._type > 0:
            return x.squeeze(self._type)
        else:
            return x.unsqueeze(-1*self._type)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Initialize dropout with the given probability
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a matrix of shape (max_len, d_model) for positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Generate position indices (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term for scaling positions in sine/cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even dimensions (0, 2, 4, ...) and cosine to odd dimensions (1, 3, 5, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add an extra dimension to the beginning to broadcast across batches later
        pe = pe.unsqueeze(0)
        
        # Register pe as a buffer so it won't be updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input sequence
        x = x + self.pe[:, :x.size(1), :]
        
        # Apply dropout to the sum of embeddings and positional encodings
        return self.dropout(x)
    
class SpecEncoder(nn.Module):
    def __init__(self,num_channels):
        super(SpecEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=3, stride=1, padding=1),  #  (num_channels, 256, 256)
            nn.MaxPool2d(kernel_size=2, stride=2),  #  (num_channels, 128, 128)
            nn.LayerNorm([num_channels,128,128]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1),  #  (num_channels*2, 128, 128)
            nn.MaxPool2d(kernel_size=4, stride=4),  #  (num_channels, 32, 32)
            nn.LayerNorm([num_channels,32,32]),
            nn.LeakyReLU(),
            Reshape(num_channels*32*32),
        )
    def forward(self, x):
        return self.layers(x)

class mmWaveBranch(nn.Module):
    def __init__(self,params):
        super(mmWaveBranch, self).__init__()
        self.num_channels = params.num_channels
        self.dropout_ = params.dropout_rate
        
        self.range_encoder = SpecEncoder(self.num_channels)
        self.music_encoder = SpecEncoder(self.num_channels)
        self.doppler_encoder = SpecEncoder(self.num_channels)

        self.fusion_block = nn.Sequential(
            nn.Linear(3*self.num_channels*32*32, 4096),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(),
        )
        
        self.position_encoder = PositionalEncoding(d_model=1024,dropout=0.1,max_len=300)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024,nhead=2,batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=3)  
    
        
    def masked_mean_pooling(self, output, mask):
        # output: (batch_size, max_len, d_model)
        # mask: (batch_size, max_len), True means padding
        mask = (~mask).float().unsqueeze(-1)  # inverse masking，True means valid data，(batch_size, max_len, 1)
        # Compute the sum of the output for each sample
        masked_output = output * mask
        sum_output = masked_output.sum(dim=1)  # (batch_size, d_model)   
        # Compute the length of each valid sample
        lengths = mask.sum(dim=1)  # (batch_size, 1) 
        # Compute the mean of the output for each valid sample
        mean_output = sum_output / lengths  
        return mean_output
    
    def forward(self,specs,specs_mask):
        """
        spec shpae: (m,n,3,256,256). m is the batch size, n is the number of frames for one sample sequence
        Given mask, padding with zero frames would not affect correctness, as long as we ensure that amp_encoder,
        amp_mapping, phase_encoder, and fusion_block do not involve cross frame operations  
        """
        range_spec = specs[:,:,0,:,:].view(-1,1,256,256)  # (m,n,256,256)->(m*n,1,256,256)
        music_spec = specs[:,:,1,:,:].view(-1,1,256,256)   
        doppler_spec = specs[:,:,2,:,:].view(-1,1,256,256)    
        range_embed = self.range_encoder(range_spec)
        music_embed = self.music_encoder(music_spec)
        doppler_embed = self.doppler_encoder(doppler_spec)
        fused_embed = torch.cat((range_embed,music_embed,doppler_embed),dim=-1)

        # fused_embed[:,dim*2:dim*3]=0.0
        fusion = self.fusion_block(fused_embed)
        fusion = fusion.view(specs.shape[0],specs.shape[1],1024)  # (m*n,1, 1024) -> (m,n,1024) 
        # # Using transformer:
        fusion = self.position_encoder(fusion)  # add positional encoding
        transformer_out = self.transformer(fusion, src_key_padding_mask=specs_mask)  # (m,n,1024) -> (m,n,1024)
        output = self.masked_mean_pooling(transformer_out, specs_mask)  # (m,n,1024) -> (m,1024)
        return output


class ActionNet(nn.Module):
    def __init__(self,params):
        super(ActionNet, self).__init__()
        self.dropout_=params.dropout_rate
        self.num_channels = params.num_channels
        self.action_num = len(params.use_action)
        # the input phase matrix is (Rx_num，adc_sample，Tx_num), and the conv1d layer here conv the phase of anntena array along range bin (adc_sample)   
        
        self.mmWaveBranch = mmWaveBranch(params)
        
        self.action_classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, self.action_num),
            nn.Softmax(dim=-1),
        )
        for m in self.modules():                           # weights initialization
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
        
    def forward(self,specs,specs_mask):
        """
        spec shpae: (m,n,3,256,256). m is the batch size, n is the number of frames for one sample sequence
        Given mask, padding with zero frames would not affect correctness, as long as we ensure that amp_encoder,
        amp_mapping, phase_encoder, and fusion_block do not involve cross frame operations  
        """
        mmWave_output = self.mmWaveBranch(specs,specs_mask)  # (m,n,1024) -> (m,1024)
        action_output = self.action_classifier(mmWave_output)  # (m,64) -> (m,64)

        return action_output

def loss_fn(action_logits, action_label):
    """
    Loss function
    """
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(action_logits,action_label)
    return loss

class SMPL_metrics():
    """
    Evaluation metrics
    """
    def __init__(self,pred_joints, target_pose, target_ori, target_transl):
        self.device = pred_joints.device
        self.joints = pred_joints
        self.sample_num = pred_joints.shape[0] 
        self.smplx_model = smplx.create('/home/hopwins/Datasets/3dModels/smplx/SMPLX_NEUTRAL.npz',use_pca=False, model_type='smplx', ext='npz', batch_size=self.sample_num).to(self.device)
        with torch.no_grad():
            self.gt_smplx = self.smplx_model(body_pose=target_pose, global_orient=target_ori, transl=target_transl)
            # self.pred_smplx = self.smplx_model(body_pose=pred_pose, global_orient=pred_ori, transl=pred_transl)
        
    def metrics(self):
        
        # Average Joint Localizetion Error
        gt_joints_loc = self.gt_smplx.joints[:,:22,:]  # 24 joints in SMPLX model
        pred_joints_loc = self.joints
        JLE = F.pairwise_distance(gt_joints_loc, pred_joints_loc, p=2,eps=1e-8)
        AJLE = JLE.mean().item()
        MedJLE = JLE.median().item()
        
        metrics = {'AJLE': AJLE,
                   'MedJLE': MedJLE,}
        
        return metrics
    
def classifer_metrics(action_logits, action_label):
    """
    Classification metrics
    """
    action_pred = torch.argmax(action_logits, dim=1).cpu().detach().numpy()
    action_label = action_label.cpu().detach().numpy()
    
    from sklearn.metrics import f1_score,accuracy_score
    action_acc = accuracy_score(action_label, action_pred)
    action_f1 = f1_score(action_label, action_pred,average='macro')
    
    metrics = {'Action Accuracy': action_acc,
               'Action F1 score': action_f1,
                }     
    return metrics

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('The total size of the model is: {:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)
 

if __name__ == '__main__':
    
    # test for model
    params = model_utils.Params("./data/exp_settings/classify/params.json")
    model = ActionNet(params)
    getModelSize(model)
    
    # 模拟输入：三个不等长的序列
    seq1 = torch.randn(5, 3,256,256)
    seq2 = torch.randn(7, 3,256,256)
    seq3 = torch.randn(3, 3,256,256)
    seq4 = torch.randn(6, 3,256,256)
    batch_data = [seq1, seq2, seq3,seq4]

    # 1. 使用 pad_sequence 补零
    padded_batch = pad_sequence(batch_data, batch_first=True)  # (batch_size, max_len, input_dim)
    print("padd shape",padded_batch.shape)
    # 2. 生成 src_key_padding_mask
    lengths = [len(seq1), len(seq2), len(seq3),len(seq4)]
    max_len = padded_batch.size(1)
    src_key_padding_mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        src_key_padding_mask[i, length:] = True  # 填充位置设置为 True
        
    output = model(padded_batch, specs_mask=src_key_padding_mask)
    print("output shape",output.shape)
    print("src_key_padding_mask shape",src_key_padding_mask.shape,src_key_padding_mask)
    loss = loss_fn(output, torch.tensor([0, 1, 1,0]))
    print("loss",loss)
    print(classifer_metrics(output, torch.tensor([0, 1, 1,0])))
    # print(pose, ori, transl)
    
    # test for AVE
    # pred_pose = torch.from_numpy(np.load("/disk1/Datasets/HOI/pose_est/1/pose.npy")[0:10])
    # pred_ori = torch.from_numpy(np.load("/disk1/Datasets/HOI/pose_est/1/orient.npy")[0:10])
    # pred_transl = torch.from_numpy(np.load("/disk1/Datasets/HOI/pose_est/1/transl.npy")[0:10])
    
    # gt_pose = torch.from_numpy(np.load("/disk1/Datasets/HOI/pose_est/1/pose.npy")[20:30])
    # gt_ori = torch.from_numpy(np.load("/disk1/Datasets/HOI/pose_est/1/orient.npy")[20:30])
    # gt_transl = torch.from_numpy(np.load("/disk1/Datasets/HOI/pose_est/1/transl.npy")[20:30])
    
    # metrics = SMPL_metrics(pred_pose, pred_ori, pred_transl, gt_pose, gt_ori, gt_transl).metrics()
    # print(metrics)
    
    # metrics['loss'] = 0.01
    # print(metrics)
    
    pass
    
    