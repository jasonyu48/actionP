import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Import all dataset classes
from FocusDataset import FocusProcessedDataset, FocusOriginalDataset, LWDataset, collate_fn, lw_collate_fn
from CustomDataset import XRFProcessedDataset

# Import all model classes
from model.resnet2d import resnet18_mutual
from model.net_model import ActionNet as FocusActionNet
from model.net_model_lw import ActionNet as LWActionNet
import model_utils

def plot_confusion_matrix(y_true, y_pred, classes, save_path, title="Confusion Matrix"):
    """
    Plot and save confusion matrix with counts.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Plot heatmap without annotations first
    ax = sns.heatmap(cm, cmap='Blues', xticklabels=classes, yticklabels=classes, annot=False)
    
    # Add text annotations manually
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Add count in the center of the cell
            ax.text(j + 0.5, i + 0.5, str(cm[i,j]),
                   ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    
    # Increase tick label sizes
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=45, va='center', fontsize=12)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot with high resolution
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_xrf_model(model_path, use_multi_angle=True, use_noisy=False):
    """
    Evaluate XRF model and generate confusion matrix.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test dataset
    test_dataset = XRFProcessedDataset(split='test', use_multi_angle=use_multi_angle)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)
    
    # Load model
    model = resnet18_mutual().to(device)
    # Set weights_only=False to handle the PyTorch 2.6 change
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Get predictions
    with torch.no_grad():
        for batch_data in test_loader:
            # Unpack correctly based on what XRFProcessedDataset returns
            if len(batch_data) == 3:
                # Old version returns (sim_data, noisy_data, labels)
                sim_data, noisy_data, labels = batch_data
            elif len(batch_data) == 4:
                # New version returns (sim_data, noisy_data, labels, angle)
                sim_data, noisy_data, labels, _ = batch_data
            else:
                raise ValueError(f"Unexpected number of return values from XRFProcessedDataset: {len(batch_data)}")
            
            # Use either noisy or sim data based on parameter
            data = noisy_data if use_noisy else sim_data
            data = data.to(device)
            outputs, _ = model(data)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Get class names from dataset
    class_names = test_dataset.action_types
    
    return all_labels, all_preds, class_names

def evaluate_focus_model(model_path, use_multi_angle=True, use_noisy=False, use_original=False):
    """
    Evaluate Focus model and generate confusion matrix.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load parameters
    # The model path is likely to be in a checkpoint directory with a params.json file
    model_dir = os.path.dirname(model_path)
    params_path = os.path.join(model_dir, 'params.json')
    
    if os.path.exists(params_path):
        params = model_utils.Params(params_path)
    else:
        # Use default parameters path if not in model directory
        params = model_utils.Params('params.json')
    
    # Load test dataset
    if use_original:
        test_dataset = FocusOriginalDataset(split='test', use_multi_angle=use_multi_angle)
    else:
        test_dataset = FocusProcessedDataset(split='test', use_multi_angle=use_multi_angle)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                             num_workers=1, collate_fn=collate_fn)
    
    # Load model
    model = FocusActionNet(params).to(device)
    # Set weights_only=False to handle the PyTorch 2.6 change
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Get predictions
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device based on data type
            if use_noisy:
                RDspecs = batch['noisy_RDspecs'].to(device)
                AoAspecs = batch['noisy_AoAspecs'].to(device)
            else:
                RDspecs = batch['sim_RDspecs'].to(device)
                AoAspecs = batch['sim_AoAspecs'].to(device)
            
            padding_mask = batch['padding_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(RDspecs, AoAspecs, padding_mask)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Get class names from dataset
    class_names = test_dataset.action_types
    
    return all_labels, all_preds, class_names

def evaluate_lw_model(model_path, use_multi_angle=True, use_noisy=False):
    """
    Evaluate LW model and generate confusion matrix.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load parameters
    # The model path is likely to be in a checkpoint directory with a params.json file
    model_dir = os.path.dirname(model_path)
    params_path = os.path.join(model_dir, 'params.json')
    
    if os.path.exists(params_path):
        params = model_utils.Params(params_path)
    else:
        # Use default parameters path if not in model directory
        params = model_utils.Params('params.json')
    
    # Load test dataset
    test_dataset = LWDataset(split='test', use_multi_angle=use_multi_angle)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                             num_workers=1, collate_fn=lw_collate_fn)
    
    # Load model
    model = LWActionNet(params).to(device)
    # Set weights_only=False to handle the PyTorch 2.6 change
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Get predictions
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device based on data type
            if use_noisy:
                specs = batch['noisy_specs'].to(device)
            else:
                specs = batch['sim_specs'].to(device)
            
            padding_mask = batch['padding_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(specs, padding_mask)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Get class names from dataset
    class_names = test_dataset.action_types
    
    return all_labels, all_preds, class_names

def parse_args():
    parser = argparse.ArgumentParser(description='Plot confusion matrices for all models')
    parser.add_argument('--lw_model_path', type=str, default='/scratch/tshu2/jyu197/XRF55-repo/lw_model_multi_angle/best.pth.tar',
                        help='Path to LW model checkpoint')
    parser.add_argument('--focus_model_path', type=str, default='/scratch/tshu2/jyu197/XRF55-repo/focus_processed_model_multiangle/best.pth.tar',
                        help='Path to Focus model checkpoint')
    parser.add_argument('--xrf_model_path', type=str, default='/scratch/tshu2/jyu197/XRF55-repo/result/params/train_mmwave_processed_noisy/model_best.pth',
                        help='Path to XRF model checkpoint')
    parser.add_argument('--use_multi_angle', action='store_true', default=True,
                        help='Use data from all angles (0, 90, 180, 270) instead of just 90 degrees')
    parser.add_argument('--use_noisy', action='store_true', default=True,
                        help='Use noisy data for evaluation instead of simulated data')
    parser.add_argument('--use_original', action='store_true', default=False,
                        help='Use original Focus dataset for evaluation instead of processed')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create results directory if it doesn't exist
    save_dir = 'confusion_matrix'
    os.makedirs(save_dir, exist_ok=True)
    
    # # Evaluate LW model
    # print("Evaluating LW model...")
    # lw_labels, lw_preds, lw_classes = evaluate_lw_model(
    #     args.lw_model_path, 
    #     use_multi_angle=args.use_multi_angle, 
    #     use_noisy=args.use_noisy
    # )
    
    # # Plot LW confusion matrix
    # lw_title = f"LW Model Confusion Matrix ({'Multi-Angle' if args.use_multi_angle else '90° Only'}, {'Noisy' if args.use_noisy else 'Sim'} Data)"
    # plot_confusion_matrix(
    #     lw_labels, 
    #     lw_preds, 
    #     lw_classes,
    #     os.path.join(save_dir, 'lw_confusion_matrix.png'),
    #     title=lw_title
    # )
    # lw_accuracy = (lw_preds == lw_labels).mean() * 100
    # print(f"LW Model Test Accuracy: {lw_accuracy:.2f}%")
    
    # # Evaluate Focus model
    # print("Evaluating Focus model...")
    # focus_labels, focus_preds, focus_classes = evaluate_focus_model(
    #     args.focus_model_path, 
    #     use_multi_angle=args.use_multi_angle, 
    #     use_noisy=args.use_noisy,
    #     use_original=args.use_original
    # )
    
    # # Plot Focus confusion matrix
    # focus_title = f"Focus Model Confusion Matrix ({'Original' if args.use_original else 'Processed'}, {'Multi-Angle' if args.use_multi_angle else '90° Only'}, {'Noisy' if args.use_noisy else 'Sim'} Data)"
    # plot_confusion_matrix(
    #     focus_labels, 
    #     focus_preds, 
    #     focus_classes,
    #     os.path.join(save_dir, 'focus_confusion_matrix.png'),
    #     title=focus_title
    # )
    # focus_accuracy = (focus_preds == focus_labels).mean() * 100
    # print(f"Focus Model Test Accuracy: {focus_accuracy:.2f}%")
    
    # Evaluate XRF model
    print("Evaluating XRF model...")
    xrf_labels, xrf_preds, xrf_classes = evaluate_xrf_model(
        args.xrf_model_path, 
        use_multi_angle=args.use_multi_angle, 
        use_noisy=args.use_noisy
    )
    
    # Plot XRF confusion matrix
    xrf_title = f"XRF Model Confusion Matrix ({'Multi-Angle' if args.use_multi_angle else '90° Only'}, {'Noisy' if args.use_noisy else 'Sim'} Data)"
    plot_confusion_matrix(
        xrf_labels, 
        xrf_preds, 
        xrf_classes,
        os.path.join(save_dir, 'xrf_confusion_matrix.png'),
        title=xrf_title
    )
    xrf_accuracy = (xrf_preds == xrf_labels).mean() * 100
    print(f"XRF Model Test Accuracy: {xrf_accuracy:.2f}%")
    
    # # Print overall comparison
    # print("\nModel Comparison:")
    # print(f"LW Model: {lw_accuracy:.2f}%")
    # print(f"Focus Model: {focus_accuracy:.2f}%")
    # print(f"XRF Model: {xrf_accuracy:.2f}%")

if __name__ == "__main__":
    main() 