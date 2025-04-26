import json
import os
import torch
import argparse
from model.net_model import ActionNet

class Params:
    """Class that loads parameters from a JSON file."""
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
            
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

def count_parameters(model):
    """
    Count the parameters of a PyTorch model and return details.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (param_count, param_size_mb, total_size_mb)
    """
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_count += param.nelement()
    
    buffer_size = 0
    buffer_count = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_count += buffer.nelement()
    
    total_size_mb = (param_size + buffer_size) / 1024 / 1024
    param_size_mb = param_size / 1024 / 1024
    
    print(f'Model Size: {total_size_mb:.2f} MB')
    print(f'Parameters: {param_count:,} ({param_size_mb:.2f} MB)')
    print(f'Buffers: {buffer_count:,} ({buffer_size / 1024 / 1024:.2f} MB)')
    
    return param_count, param_size_mb, total_size_mb

def main():
    parser = argparse.ArgumentParser(description='Count parameters of a trained model')
    parser.add_argument('--model_dir', type=str, default='./focus_processed_model_multiangle', 
                        help='Directory containing model checkpoint and params.json')
    args = parser.parse_args()
    
    # Import json here to avoid circular import
    import json
    
    # Load parameters
    params_path = os.path.join(args.model_dir, 'params.json')
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No params.json found in {args.model_dir}")
    
    params = Params(params_path)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() and params.cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with same architecture
    model = ActionNet(params)
    model = model.to(device)
    
    # Count parameters before loading weights
    param_count, param_size_mb, total_size_mb = count_parameters(model)
    print(f"\nModel has {param_count:,} trainable parameters")
    
    # Try to load checkpoint if available (not required for parameter counting)
    checkpoint_path = os.path.join(args.model_dir, 'best.pth.tar')
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print("Checkpoint loaded successfully")
            print(f"Trained model validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Parameter count is still accurate as it's based on the model architecture")
    
if __name__ == "__main__":
    main() 