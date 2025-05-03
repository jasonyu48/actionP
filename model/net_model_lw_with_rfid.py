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
    
    def forward(self,mm_data_list):
        """
        spec shpae: (m,n,3,256,256). m is the batch size, n is the number of frames for one sample sequence
        Given mask, padding with zero frames would not affect correctness, as long as we ensure that amp_encoder,
        amp_mapping, phase_encoder, and fusion_block do not involve cross frame operations  
        """
        specs, specs_mask = mm_data_list
        range_spec = specs[:,:,0,:,:].view(-1,1,256,256)  # (m,n,256,256)->(m*n,1,256,256)
        music_spec = specs[:,:,1,:,:].view(-1,1,256,256)   
        doppler_spec = specs[:,:,2,:,:].view(-1,1,256,256)    
        range_embed = self.range_encoder(range_spec)
        music_embed = self.music_encoder(music_spec)
        doppler_embed = self.doppler_encoder(doppler_spec)
        fused_embed = torch.cat((range_embed,music_embed,doppler_embed),dim=-1)
        fusion = self.fusion_block(fused_embed)
        fusion = fusion.view(specs.shape[0],specs.shape[1],1024)  # (m*n,1, 1024) -> (m,n,1024) 
        # # Using transformer:
        fusion = self.position_encoder(fusion)  # add positional encoding
        transformer_out = self.transformer(fusion, src_key_padding_mask=specs_mask)  # (m,n,1024) -> (m,n,1024)
        output = self.masked_mean_pooling(transformer_out, specs_mask)  # (m,n,1024) -> (m,1024)
        return output

class OBJBranch(nn.Module):
    def __init__(self,params):
        super(OBJBranch, self).__init__()
        self.dropout_ = params.dropout_rate
        self.obj_num = params.obj_num
        self.neuron_num = params.obj_branch_size
        self.obj_dim = params.obj_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.obj_dim, self.neuron_num//2),
            nn.LeakyReLU(),
            nn.Linear(self.neuron_num//2, self.neuron_num),
            nn.LeakyReLU(),
        )
        self.position_encoder = PositionalEncoding(d_model=self.neuron_num,dropout=0.1,max_len=300)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.neuron_num,nhead=2,batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=3)  
        self.embed_head =  nn.Linear(300, self.neuron_num)
        self.mapping = nn.Linear(self.obj_num*self.neuron_num*2, 1024)
        
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

    def forward(self,rfid_data_list):
        # input obj_loc shape: (m,n,6,4), m is the batch size, n is the number of frames for one sample sequence
        # note that obj_loc can be the phase information, not the 3d coordinate
        # obj_mask shape: (m,n,6), True means padding
        # obj_name shape: (m,6,300), the object name embeddings for each object 

        obj_loc, obj_mask, obj_name = rfid_data_list
        # (m,n,6,4)->(m,6,n,4)
        obj_embed = obj_loc.permute(0,2,1,3)
        obj_embed = obj_embed[:,:,:,:self.obj_dim]
        # (m,6,n,4)->(m*6,n,4)
        obj_embed = obj_embed.reshape(-1,obj_embed.shape[-2],obj_embed.shape[-1])
        obj_embed = self.encoder(obj_embed)  # (m*6,n,4) -> (m*6,n,64)
        
        obj_mask = obj_mask.repeat(self.obj_num,1)
        # Using transformer:
        obj_embed = self.position_encoder(obj_embed)   # (m*6,n,64) -> (m*6,n,64)
        transformer_out = self.transformer(obj_embed, src_key_padding_mask=obj_mask) # (m*6,n,64) -> (m*6,n,64)
        output = self.masked_mean_pooling(transformer_out, obj_mask) # (m*6,n,64) -> (m*6,64)
        # output = transformer_out[:,0,:] # (m*6,n,64) -> (m*6,64)
        
        obj_embedding = self.embed_head(obj_name)  # (m,6,300) -> (m,6,64)
        obj_embedding = obj_embedding.reshape(-1,self.neuron_num)  # (m,6,64) -> (m*6,64)
        output = torch.cat((output,obj_embedding),dim=-1)  # (m*6,64) + (m*6,64) -> (m*6,128)
        output = output.reshape(obj_loc.shape[0],self.obj_num*self.neuron_num*2)  # (m*6,64) -> (m,6*64*2)
        output = self.mapping(output)  # (m,6*64) -> (m,1024)
        return output
        

class HOINet(nn.Module):
    def __init__(self,params):
        super(HOINet, self).__init__()
        self._input_dim = params.input_dim
        self.dropout_=params.dropout_rate
        self.num_channels = params.num_channels
        self.action_num = len(params.use_action)
        self.obj_num = params.obj_num
        # the input phase matrix is (Rx_num，adc_sample，Tx_num), and the conv1d layer here conv the phase of anntena array along range bin (adc_sample)   
        
        self.mmWaveBranch = mmWaveBranch(params)
        self.OBJBranch = OBJBranch(params)
        self.action_classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, self.action_num),
        )
        self.obj_classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, self.obj_num),
        )
        for m in self.modules():                           # weights initialization
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
        
    def forward(self,mm_data_list,rfid_data_list):
        """
        spec shpae: (m,n,3,256,256). m is the batch size, n is the number of frames for one sample sequence
        Given mask, padding with zero frames would not affect correctness, as long as we ensure that amp_encoder,
        amp_mapping, phase_encoder, and fusion_block do not involve cross frame operations  
        """
        mmWave_output = self.mmWaveBranch(mm_data_list)  # (m,n,1024) -> (m,1024)
        obj_output = self.OBJBranch(rfid_data_list)  # (m,n,6,3) -> (m,1024)
        
        fuse_output = torch.cat((mmWave_output,obj_output),dim=-1)  # (m,1024) + (m,1024) -> (m,2048)
        # fuse_output = mmWave_output + obj_output # (m,1024) + (m,1024) -> (m,1024)


        # action_output = self.action_classifier(obj_output)  # (m,64) -> (m,64)
        # objid_output = self.obj_classifier(obj_output)  # (m,1024) -> (m,6)
        action_output = self.action_classifier(fuse_output)  # (m,64) -> (m,64)
        objid_output = self.obj_classifier(fuse_output)  # (m,1024) -> (m,6)

        return action_output, objid_output


    
def loss_fn(action_logits, action_label,obj_logits, obj_label,params):
    """
    Loss function. 
    params.loss_coef: the weight of the object classification loss
    """
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(action_logits,action_label) + params.loss_coef * loss_fn(obj_logits, obj_label)
    return loss

def classifer_metrics(action_logits, action_label,obj_logits, obj_label):
    """
    Classification metrics
    """
    action_pred = torch.argmax(action_logits, dim=1).cpu().detach().numpy()
    action_label = action_label.cpu().detach().numpy()
    
    obj_pred = torch.argmax(obj_logits, dim=1).cpu().detach().numpy()
    obj_label = obj_label.cpu().detach().numpy()
    
    from sklearn.metrics import f1_score,accuracy_score
    action_acc = accuracy_score(action_label, action_pred)
    action_f1 = f1_score(action_label, action_pred,average='macro')
    
    obj_acc = accuracy_score(obj_label, obj_pred)
    obj_f1 = f1_score(obj_label, obj_pred,average='macro')
    
    metrics = {'Action Accuracy': action_acc,
               'Action F1 score': action_f1,
               'Object Accuracy': obj_acc,
               'Object F1 score': obj_f1}
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
 
def getDebugData():
    # 模拟输入：四个不等长的序列
    seq1 = torch.randn(5, 3,256,256)
    seq2 = torch.randn(7, 3,256,256)
    seq3 = torch.randn(3, 3,256,256)
    seq4 = torch.randn(6, 3,256,256)
    batch_data = [seq1, seq2, seq3,seq4]
    # 1. 使用 pad_sequence 补零
    padded_mmwave_batch = pad_sequence(batch_data, batch_first=True)  # (batch_size, max_len, input_dim)
    print("padd shape",padded_mmwave_batch.shape)
    # 2. 生成 src_key_padding_mask
    lengths = [len(seq1), len(seq2), len(seq3),len(seq4)]
    max_len = padded_mmwave_batch.size(1)
    action_pad_mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        action_pad_mask[i, length:] = True  # 填充位置设置为 True
    
    # obj location batch: (batch_size, max_len, 3)
    seq1 = torch.randn(7, 6, 4)
    seq2 = torch.randn(9, 6, 4)
    seq3 = torch.randn(11, 6, 4)
    seq4 = torch.randn(13, 6, 4)
    batch_data = [seq1, seq2, seq3,seq4]
    # 1. 使用 pad_sequence 补零
    padded_obj_batch = pad_sequence(batch_data, batch_first=True)  # (batch_size, max_len, input_dim)
        # 2. 生成 src_key_padding_mask
    lengths = [len(seq1), len(seq2), len(seq3),len(seq4)]
    max_len = padded_obj_batch.size(1)
    obj_pad_mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        obj_pad_mask[i, length:] = True  # 填充位置设置为 True

    obj_name = torch.ones((4,6,300))
    return [padded_mmwave_batch, action_pad_mask], [padded_obj_batch, obj_pad_mask,obj_name]


if __name__ == '__main__':
    
    # test for model
    params = model_utils.Params("./data/exp_settings/hoi/params.json")
    model = HOINet(params)
    getModelSize(model)
    
    
    mm_data_list, rfid_data_list = getDebugData()
    
    action_output, obj_output = model(mm_data_list, rfid_data_list)
    print("output shape",action_output.shape,obj_output.shape)
    print("src_key_padding_mask shape",mm_data_list[1].shape,rfid_data_list[1].shape)
    loss = loss_fn(action_output, torch.tensor([0, 1, 1,0]),obj_output, torch.tensor([0, 2, 1,4]),params)
    print("loss",loss)
    print(classifer_metrics(action_output, torch.tensor([0, 1, 1,0]),obj_output, torch.tensor([0, 2, 1,4])))

    
    