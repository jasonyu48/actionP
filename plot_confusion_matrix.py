import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from CustomDataset import XRFProcessedDataset
from model.resnet2d import resnet18_mutual

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
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
    
    plt.title('Confusion Matrix', fontsize=16, pad=20)
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

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test dataset
    test_dataset = XRFProcessedDataset(base_dir='./xrf555_processed', split='test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)
    
    # Load model
    model = resnet18_mutual().to(device)
    model.load_state_dict(torch.load('/scratch/tshu2/jyu197/XRF55-repo/result/params/train_mmwave_processed_sim/model_best.pth'))
    model.eval()
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Get predictions
    with torch.no_grad():
        for sim_data, noisy_data, labels in test_loader:
            # Use sim_data as we're using the sim model
            sim_data = sim_data.to(device)
            outputs, _ = model(sim_data)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Get class names from dataset
    class_names = test_dataset.action_types
    
    # Create results directory if it doesn't exist
    save_dir = 'result/confusion_matrix'
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(
        all_labels, 
        all_preds, 
        class_names,
        os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    # Calculate and print overall accuracy
    accuracy = (all_preds == all_labels).mean() * 100
    print(f"Overall Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main() 