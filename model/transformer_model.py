import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import math
import json

# Global constants
ENCODER_DIM = 256  # Dimension for all encoder models
FUSION_DIM = 512  # Dimension for final fusion transformer (was 512) which is the best

class Reshape(torch.nn.Module):
    def __init__(self, type):
        super(Reshape, self).__init__()
        self._type = type

    def forward(self, x):
        if self._type == -1:
            return x.permute(0, 2, 1, 3)
        else:
            return x.view(-1, self._type)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000): #was max_len=5000, best is max_len=300
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
    def __init__(self, d_model=ENCODER_DIM, nhead=4, dropout=0.1, num_antennas=12):
        super(RadarEncoderTransformer, self).__init__()
        
        # Reshape inputs - project from antenna*range dimensions to d_model
        self.input_projection = nn.Linear(32*num_antennas, d_model)
        
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Global pooling over sequence - using mean pooling followed by projection
        self.global_pooling = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256)
        )
        
    def forward(self, x):
        # x shape: (-1, num_antennas, 128, 32)
        batch_size = x.size(0)
        
        # Reshape to (-1, 128, num_antennas*32)
        x = x.permute(0, 2, 1, 3)  # -> (-1, 128, num_antennas, 32)
        x = x.reshape(batch_size, 128, -1)  # -> (-1, 128, num_antennas*32)
        
        # Project to d_model dimensions
        x = self.input_projection(x)  # -> (-1, 128, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # -> (-1, 128, d_model)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # -> (-1, 128, d_model)
        
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x):
        # x shape: (-1, 1, 256, 256)
        batch_size = x.size(0)
        
        # Extract patches and project - from (-1, 1, 256, 256) to (-1, d_model, patches, patches)
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
    def __init__(self, dropout_rate=0.1, num_antennas=12, data_format='processed'):
        super(mmWaveBranch, self).__init__()
        self.dropout_ = dropout_rate
        self.num_antennas = num_antennas
        self.data_format = data_format  # 'processed' or 'original'
        
        # For processed Focus data, use transformer-based encoders
        self.range_encoder = RadarEncoderTransformer(
            d_model=ENCODER_DIM,
            nhead=4,
            dropout=self.dropout_,
            num_antennas=self.num_antennas
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
            num_antennas=self.num_antennas
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
        
        self.position_encoder = PositionalEncoding(d_model=FUSION_DIM, dropout=self.dropout_, max_len=300)
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
    
    def forward(self, RDspecs, AoAspecs, specs_mask):
        """
        Process data in the processed Focus format:
        
        RDspecs: (batch_size, time_steps, 2, num_antennas, 128, 32) 
                where 32 = 31 (range data) + 1 (normalized position)
        AoAspecs: (batch_size, time_steps, 256, 256)
        specs_mask: (batch_size, time_steps) padding mask
        """
        batch_size, time_steps = RDspecs.shape[0], RDspecs.shape[1]
        
        # Process range data (dim=0)
        range_data = RDspecs[:, :, 0]  # (batch_size, time_steps, num_antennas, 128, 32)
        range_data = range_data.view(-1, self.num_antennas, 128, 32)
        range_embed = self.range_encoder(range_data)
        
        # Process AoA data for music encoder
        aoa_data = AoAspecs.view(-1, 1, 256, 256)  # (batch_size*time_steps, 1, 256, 256)
        music_embed = self.music_encoder(aoa_data)
        
        # Process doppler data (dim=1)
        doppler_data = RDspecs[:, :, 1]  # (batch_size, time_steps, num_antennas, 128, 32)
        doppler_data = doppler_data.view(-1, self.num_antennas, 128, 32)
        doppler_embed = self.doppler_encoder(doppler_data)
        
        # Concatenate all embeddings
        fused_embed = torch.cat((range_embed, music_embed, doppler_embed), dim=-1)
        
        # Process through fusion block
        fusion = self.fusion_block(fused_embed)
        fusion = fusion.view(batch_size, time_steps, FUSION_DIM)  # Reshape back to (batch_size, time_steps, FUSION_DIM)
        
        # Apply positional encoding and transformer
        fusion = self.position_encoder(fusion)
        transformer_out = self.transformer(fusion, src_key_padding_mask=specs_mask)
        output = self.masked_mean_pooling(transformer_out, specs_mask)
        
        return output

class TransformerActionNet(nn.Module):
    def __init__(self, action_num, dropout_rate=0.1, num_antennas=12, data_format='processed'):
        super(TransformerActionNet, self).__init__()
        self.dropout_ = dropout_rate
        self.action_num = action_num
        
        self.mmWaveBranch = mmWaveBranch(
            dropout_rate=dropout_rate,
            num_antennas=num_antennas,
            data_format=data_format
        )
        
        self.action_classifier = nn.Sequential(
            nn.Linear(FUSION_DIM, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_),
            nn.Linear(64, self.action_num),
            nn.Softmax(dim=-1),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
        
    def forward(self, RDspecs, AoAspecs, specs_mask):
        """
        Process Focus_processed data:
        RDspecs: (batch_size, time_steps, 2, num_antennas, 128, 32) 
                where 32 = 31 (range data) + 1 (normalized position)
        AoAspecs: (batch_size, time_steps, 256, 256)
        specs_mask: (batch_size, time_steps) padding mask
        """
        mmWave_output = self.mmWaveBranch(RDspecs, AoAspecs, specs_mask)
        action_output = self.action_classifier(mmWave_output)
        return action_output

def loss_fn(action_logits, action_label):
    """
    Loss function
    """
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(action_logits, action_label)
    return loss

def classifer_metrics(action_logits, action_label):
    """
    Classification metrics
    """
    action_pred = torch.argmax(action_logits, dim=1).cpu().detach().numpy()
    action_label = action_label.cpu().detach().numpy()
    
    from sklearn.metrics import f1_score, accuracy_score
    action_acc = accuracy_score(action_label, action_pred)
    action_f1 = f1_score(action_label, action_pred, average='macro')
    
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

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

if __name__ == '__main__':
    # Test the model
    print(f"Using encoder dimension: {ENCODER_DIM}, fusion dimension: {FUSION_DIM}")
    
    # Create model with direct parameters (no params file dependency)
    action_num = 7  # Example number of action classes
    model = TransformerActionNet(
        action_num=action_num,
        dropout_rate=0.2,
        num_antennas=12,
        data_format='processed'
    )
    model_size = getModelSize(model)
    
    print(f"Model parameters: {model_size[1]:,}")
    print(f"Model size in MB: {model_size[-1]:.3f}")
    
    # Create dummy input for testing
    batch_size = 2
    time_steps = 5
    num_antennas = 12
    
    # Create dummy input data
    RDspecs = torch.randn(batch_size, time_steps, 2, num_antennas, 128, 32)
    AoAspecs = torch.randn(batch_size, time_steps, 256, 256)
    specs_mask = torch.zeros((batch_size, time_steps), dtype=torch.bool)
    specs_mask[0, 3:] = True  # Padding in the first sample
    
    # Forward pass
    outputs = model(RDspecs, AoAspecs, specs_mask)
    print(f"Output shape: {outputs.shape}") 