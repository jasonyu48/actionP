import torch
import numpy as np
from model.network_cube_learn import RDAT_3DCNNLSTM
from cplxmodule.nn.modules.casting import TensorToCplx

def test_rdat_model_variable_length():
    """
    Test RDAT_3DCNNLSTM model with variable length input sequences
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model and move it to device
    model = RDAT_3DCNNLSTM()
    model = model.to(device)
    
    # Test with different sequence lengths
    batch_size = 16
    for time_steps in [5, 45]:
        print(f"Testing with sequence length: {time_steps}")
        
        # Generate random complex data with shape [batch_size, time_steps, 128, 12, 256]
        # First create real and imaginary parts separately
        real_part = torch.randn(batch_size, time_steps, 128, 12, 256)
        imag_part = torch.randn(batch_size, time_steps, 128, 12, 256)
        
        # Combine into complex tensor format that cplxmodule can use
        # Stack real and imaginary parts along the last dimension
        complex_tensor = torch.stack([real_part, imag_part], dim=-1)
        
        # Move tensor to device before conversion
        complex_tensor = complex_tensor.to(device)
        
        # Convert to cplxmodule format using TensorToCplx
        complex_input = TensorToCplx()(complex_tensor)
        
        # Forward pass through the model
        with torch.no_grad():  # Disable gradient calculation for testing
            output = model(complex_input)
        print(f"Success! Output shape: {output.shape}")
        # Should be [batch_size, 7] regardless of input time_steps
        assert output.shape == torch.Size([batch_size, 7])

if __name__ == "__main__":
    print("Testing RDAT_3DCNNLSTM model with variable length sequences...")
    test_rdat_model_variable_length()
    print("Test completed!") 