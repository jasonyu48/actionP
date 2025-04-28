import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
import json

# Global constants
ENCODER_DIM = 256  # Dimension for mmWave branch
FUSION_DIM = 512  # Dimension for fusion transformer
NEURON_NUM = 64   # Dimension for object branch

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
    def __init__(self, d_model, dropout=0.1, max_len=512):
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
    
class RadarEncoderTransformer(nn.Module):
    """
    Transformer encoder for radar data (both range and doppler).
    This unified encoder processes a sequence of dimensions (time/velocity)
    with features from antennas and range bins.
    """
    def __init__(self, d_model=ENCODER_DIM, nhead=4, dropout=0.2):
        super(RadarEncoderTransformer, self).__init__()
        
        # Reshape inputs
        self.input_projection = nn.Linear(256, d_model)
        
        # Positional encoding for sequence dimension
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Create transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,  # Standard 4x multiplier
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1) #was num_layers=1, which is the best
        
        # Global pooling over sequence - using mean pooling followed by projection
        self.global_pooling = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )
        
    def forward(self, x):
        # x shape: (-1, 256, 256)
        
        # Project to d_model dimensions
        x = self.input_projection(x)  # -> (-1, 256, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # -> (-1, 256, d_model)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # -> (-1, 256, d_model)
        
        # Apply mean pooling over sequence dimension
        x = torch.mean(x, dim=1)  # -> (-1, d_model)
        
        # Apply projection layers
        x = self.global_pooling(x)  # -> (-1, 256)
        
        return x

class MusicEncoderViT(nn.Module):
    def __init__(self, d_model=ENCODER_DIM, nhead=4, dropout=0.1, patch_size=16):
        super(MusicEncoderViT, self).__init__()
        
        # Calculate number of patches
        self.patch_size = patch_size
        self.num_patches = (256 // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=1, 
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,  # Standard 4x multiplier
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1) #was num_layers=1, which is the best
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        # x shape: (-1, 1, 256, 256)
        batch_size = x.size(0)
        
        # Extract patches and project - from (-1, 1, 256, 256) to (-1, d_model, 256/patch_size, 256/patch_size)
        x = self.patch_embedding(x)  # -> (-1, d_model, 256/patch_size, 256/patch_size)
        
        # Reshape to sequence of patches - from (-1, d_model, patches, patches) to (-1, patches*patches, d_model)
        x = x.flatten(2).transpose(1, 2)  # -> (-1, patches*patches, d_model)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # -> (-1, 1+patches*patches, d_model)
        
        # Add position embedding
        x = x + self.pos_embedding  # -> (-1, 1+patches*patches, d_model)
        
        # Apply transformer
        x = self.transformer_encoder(x)  # -> (-1, 1+patches*patches, d_model)
        
        # Use class token for output
        x = x[:, 0]  # -> (-1, d_model)
        
        # Apply MLP head
        x = self.mlp_head(x)  # -> (-1, 256)
        
        return x

class mmWaveBranch(nn.Module):
    def __init__(self, dropout_rate=0.1, data_format='processed'):
        super(mmWaveBranch, self).__init__()
        self.dropout_ = dropout_rate
        self.data_format = data_format  # 'processed' or 'original'
        
        # For processed Focus data, use transformer-based encoders
        self.range_encoder = RadarEncoderTransformer(
            d_model=ENCODER_DIM,
            nhead=4,
            dropout=self.dropout_,
        )
        
        self.music_encoder = MusicEncoderViT(
            d_model=ENCODER_DIM,
            nhead=4,
            dropout=self.dropout_,
            patch_size=16
        )
        
        self.doppler_encoder = RadarEncoderTransformer(
            d_model=ENCODER_DIM,
            nhead=4,
            dropout=self.dropout_,
        )
        
        # Output sizes of our encoders
        range_out_size = 256
        music_out_size = 256
        doppler_out_size = 256
        
        total_fusion_size = range_out_size + music_out_size + doppler_out_size
        
        self.fusion_block = nn.Sequential(
            nn.Linear(total_fusion_size, total_fusion_size), #was nn.Linear(total_fusion_size, total_fusion_size) which is the best
            nn.LayerNorm(total_fusion_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_),
            nn.Linear(total_fusion_size, FUSION_DIM),
            nn.LayerNorm(FUSION_DIM),
            nn.LeakyReLU(),
        )
        
        self.position_encoder = PositionalEncoding(d_model=FUSION_DIM, dropout=self.dropout_, max_len=512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=FUSION_DIM, nhead=min(8, FUSION_DIM//8), dim_feedforward=FUSION_DIM*4, dropout=self.dropout_, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2) #was num_layers=2 which is the best
    
    def masked_mean_pooling(self, output, mask):
        # output: (batch_size, max_len, d_model)
        # mask: (batch_size, max_len), True means padding
        mask = (~mask).float().unsqueeze(-1)  # inverse masking，True means valid data
        # Compute the sum of the output for each sample
        masked_output = output * mask
        sum_output = masked_output.sum(dim=1)  
        # Compute the length of each valid sample
        lengths = mask.sum(dim=1)  
        # Compute the mean of the output for each valid sample
        mean_output = sum_output / lengths  
        return mean_output
    
    def forward(self, range_specs, doppler_specs, music_specs, specs_mask):
        """
        Process data in the processed Focus format:
        range_specs: (batch_size, time_steps, 256, 256)
        doppler_specs: (batch_size, time_steps, 256, 256)
        music_specs: (batch_size, time_steps, 256, 256)
        specs_mask: (batch_size, time_steps) padding mask
        """
        batch_size, time_steps = range_specs.shape[0], range_specs.shape[1]
        
        # Process range data
        range_data = range_specs.view(batch_size*time_steps, 256, 256)
        range_embed = self.range_encoder(range_data) #(batch_size*time_steps, 256)
        
        doppler_data = doppler_specs.view(batch_size*time_steps, 256, 256)
        doppler_embed = self.doppler_encoder(doppler_data) #(batch_size*time_steps, 256)

        aoa_data = music_specs.view(batch_size*time_steps, 256, 256)
        aoa_data = aoa_data.unsqueeze(1)  # Add channel dimension (batch_size*time_steps, 1, 256, 256)
        music_embed = self.music_encoder(aoa_data) #(batch_size*time_steps, 256)
        
        # Concatenate all embeddings
        fused_embed = torch.cat((range_embed, music_embed, doppler_embed), dim=-1)
        
        # Process through fusion block
        fusion = self.fusion_block(fused_embed)
        fusion = fusion.view(batch_size, time_steps, FUSION_DIM)  # Reshape back to (batch_size, time_steps, FUSION_DIM)
        
        # Apply positional encoding and transformer
        fusion = self.position_encoder(fusion)
        transformer_out = self.transformer(fusion, src_key_padding_mask=specs_mask)
        output = self.masked_mean_pooling(transformer_out, specs_mask) #(batch_size, FUSION_DIM)
        
        return output

class OBJBranch(nn.Module):
    def __init__(self, dropout_rate=0.1, obj_num=6):
        super(OBJBranch, self).__init__()
        self.dropout_ = dropout_rate
        self.obj_num = obj_num
        
        # Define the encoder for the combined object features
        # Now handling both RFID data (4 features) and object one-hot encodings (8 features)
        # Total features per object: 4 + 8 = 12
        self.encoder = nn.Sequential(
            nn.Linear(self.obj_num * (4 + 8), NEURON_NUM),  # 6 objects × (4 RFID features + 8 one-hot dims) = 72
            nn.LayerNorm(NEURON_NUM),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_),
        )
        
        # Position encoder and transformer layers
        self.position_encoder = PositionalEncoding(d_model=NEURON_NUM, dropout=self.dropout_, max_len=512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=NEURON_NUM, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2) #was num_layers=2
        
    def masked_mean_pooling(self, output, mask):
        # output: (batch_size, max_len, d_model)
        # mask: (batch_size, max_len), True means padding
        mask = (~mask).float().unsqueeze(-1)  # inverse masking，True means valid data
        # Compute the sum of the output for each sample
        masked_output = output * mask
        sum_output = masked_output.sum(dim=1)  
        # Compute the length of each valid sample
        lengths = mask.sum(dim=1)  
        # Compute the mean of the output for each valid sample
        mean_output = sum_output / lengths  
        return mean_output

    def forward(self, rfid_data_list):
        # input obj_loc shape: (m,n,6,12), m is the batch size, n is the number of frames for one sample sequence
        # where 12 = 4 (RFID data) + 8 (one-hot object encoding)
        # obj_mask shape: (m,n), True means padding
        
        obj_loc, obj_mask = rfid_data_list[:2]  # Only take the first two elements
        batch_size, seq_len = obj_loc.shape[0], obj_loc.shape[1]
        
        # Reshape obj_loc to combine all objects into a single feature vector
        # (m,n,6,12) -> (m,n,6*12)
        obj_features = obj_loc.reshape(batch_size, seq_len, 6*(4+8))
        
        # Encode the combined object features
        obj_embed = self.encoder(obj_features)  # (m,n,6*12) -> (m,n,NEURON_NUM)
        
        # Apply positional encoding
        obj_embed = self.position_encoder(obj_embed)  # (m,n,NEURON_NUM) -> (m,n,NEURON_NUM)
        
        # Apply transformer with n as sequence dimension
        transformer_out = self.transformer(obj_embed, src_key_padding_mask=obj_mask)  # (m,n,NEURON_NUM) -> (m,n,NEURON_NUM)
        
        # Apply masked mean pooling over sequence dimension
        output = self.masked_mean_pooling(transformer_out, obj_mask)  # (m,n,NEURON_NUM) -> (m,NEURON_NUM)
        
        return output
        

class HOINet(nn.Module):
    def __init__(self, action_num=64, obj_num=6, dropout_rate=0.1, data_format='processed'):
        super(HOINet, self).__init__()
        self.dropout_ = dropout_rate
        self.action_num = action_num
        self.obj_num = obj_num
        
        self.mmWaveBranch = mmWaveBranch(
            dropout_rate=dropout_rate,
            data_format=data_format
        )
        
        self.OBJBranch = OBJBranch(
            dropout_rate=dropout_rate,
            obj_num=obj_num
        )
        
        self.action_classifier = nn.Sequential(
            nn.Linear(FUSION_DIM+NEURON_NUM, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, self.action_num),
            nn.Softmax(dim=-1),
        )
        
        self.obj_classifier = nn.Sequential(
            nn.Linear(FUSION_DIM+NEURON_NUM, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, self.obj_num),
            nn.Softmax(dim=-1),
        )
        
        for m in self.modules():                           # weights initialization
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
        
    def forward(self, mmWave_data_list, rfid_data_list):
        """
        mmWave_data_list: [mmWave_data, specs_mask]
        mmWave_data: tensor of shape (batch_size, time_steps, 3, 256, 256)
        specs_mask: tensor of shape (batch_size, time_steps) padding mask
        
        rfid_data_list: [obj_loc, obj_mask, ...]
          - obj_loc: (batch_size, time_steps, obj_num, 12)
          - obj_mask: (batch_size, time_steps) padding mask
        """
        # Unpack mmWave data
        mmWave_data = mmWave_data_list[0]
        range_specs = mmWave_data[:,:,0]  # (batch_size, time_steps, 256, 256)
        doppler_specs = mmWave_data[:,:,1]  # (batch_size, time_steps, 256, 256)
        music_specs = mmWave_data[:,:,2]  # (batch_size, time_steps, 256, 256)
        specs_mask = mmWave_data_list[1]
        
        # Process through branches
        mmWave_output = self.mmWaveBranch(range_specs, doppler_specs, music_specs, specs_mask) #(batch_size, FUSION_DIM)
        obj_output = self.OBJBranch(rfid_data_list)  # (m,n,6,12) -> (m,NEURON_NUM)
        
        fuse_output = torch.cat((mmWave_output, obj_output), dim=-1)

        action_output = self.action_classifier(fuse_output)
        objid_output = self.obj_classifier(fuse_output)

        return action_output, objid_output


    
def loss_fn(action_logits, action_label, obj_logits, obj_label, loss_coef=0.5):
    """
    Loss function. 
    loss_coef: the weight of the object classification loss
    """
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(action_logits, action_label) + loss_coef * loss_fn(obj_logits, obj_label)
    return loss

def classifer_metrics(action_logits, action_label, obj_logits, obj_label):
    """
    Classification metrics
    """
    action_pred = torch.argmax(action_logits, dim=1).cpu().detach().numpy()
    action_label = action_label.cpu().detach().numpy()
    
    obj_pred = torch.argmax(obj_logits, dim=1).cpu().detach().numpy()
    obj_label = obj_label.cpu().detach().numpy()
    
    from sklearn.metrics import f1_score, accuracy_score
    action_acc = accuracy_score(action_label, action_pred)
    action_f1 = f1_score(action_label, action_pred, average='macro')
    
    obj_acc = accuracy_score(obj_label, obj_pred)
    obj_f1 = f1_score(obj_label, obj_pred, average='macro')
    
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
    batch_size = 4
    time_steps = 7
    
    # Create separate tensors for each spectrogram type with correct shapes
    range_specs = torch.randn(batch_size, time_steps, 256, 256)
    doppler_specs = torch.randn(batch_size, time_steps, 256, 256)
    music_specs = torch.randn(batch_size, time_steps, 256, 256)
    
    # Combine into a tensor of shape (batch_size, time_steps, 3, 256, 256)
    mmwave_data = torch.stack([range_specs, doppler_specs, music_specs], dim=2)
    
    # Create specs_mask: (batch_size, time_steps)
    specs_mask = torch.zeros((batch_size, time_steps), dtype=torch.bool)
    
    # Add some padding in the first two samples
    specs_mask[0, 5:] = True
    specs_mask[1, 6:] = True
    
    # Create object data
    obj_loc = torch.randn(batch_size, time_steps, 6, 12)  # (batch_size, time_steps, obj_num, obj_dim)
    obj_mask = specs_mask.clone()  # Use the same mask for simplicity
    
    # Return the data in the expected format
    mmwave_data_list = [mmwave_data, specs_mask]
    rfid_data_list = [obj_loc, obj_mask]
    return mmwave_data_list, rfid_data_list


if __name__ == '__main__':
    try:
        print("Creating model...")
        model = HOINet(
            action_num=64,
            obj_num=6,
            dropout_rate=0.2,
            data_format='processed'
        )
        
        print("Model architecture:")
        print(model)
        
        print("\nModel size:")
        getModelSize(model)
        
        print("\nGenerating test data...")
        mm_data, rfid_data_list = getDebugData()
        
        # Print shapes of input data
        range_specs, doppler_specs, music_specs = mm_data[0][:,:,0], mm_data[0][:,:,1], mm_data[0][:,:,2]
        obj_loc, obj_mask = rfid_data_list
        
        print("\nInput shapes:")
        print(f"- range_specs: {range_specs.shape}")
        print(f"- doppler_specs: {doppler_specs.shape}")
        print(f"- music_specs: {music_specs.shape}")
        print(f"- obj_loc: {obj_loc.shape}")
        print(f"- obj_mask: {obj_mask.shape}")
        
        print("\nRunning forward pass...")
        action_output, obj_output = model(mm_data, rfid_data_list)
        
        print("\nOutput shapes:")
        print(f"- action_output: {action_output.shape}")
        print(f"- obj_output: {obj_output.shape}")
        
        print("\nCalculating loss...")
        loss = loss_fn(
            action_output, 
            torch.tensor([0, 1, 1, 0]), 
            obj_output, 
            torch.tensor([0, 2, 1, 4])
        )
        print(f"Loss: {loss.item()}")
        
        print("\nCalculating metrics:")
        metrics = classifer_metrics(
            action_output, 
            torch.tensor([0, 1, 1, 0]), 
            obj_output, 
            torch.tensor([0, 2, 1, 4])
        )
        for key, value in metrics.items():
            print(f"- {key}: {value:.4f}")
            
        print("\nTest completed successfully!")
        
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        print(traceback.format_exc())

    
    