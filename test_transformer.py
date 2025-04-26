import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

# Import custom modules
from model.transformer_model import TransformerActionNet, loss_fn, classifer_metrics, getModelSize
from FocusDataset import FocusProcessedDataset, collate_fn
import model_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Transformer_Testing')

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Action Recognition Testing')
    
    # Model parameters
    parser.add_argument('--encoder_dim', type=int, default=256, 
                        help='Dimension of encoder models')
    parser.add_argument('--num_antennas', type=int, default=12, 
                        help='Number of antennas in radar data')
    parser.add_argument('--dropout_rate', type=float, default=0.2, 
                        help='Dropout rate for all models')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./Focus_processed_multi_angle', 
                        help='Directory of data')
    parser.add_argument('--use_multi_angle', action='store_true', default=True, 
                        help='Use data from all angles (0, 90, 180, 270) instead of just 90 degrees')
    parser.add_argument('--use_noisy', action='store_true', default=True, 
                        help='Use noisy data for inference instead of simulated data')
    
    # Inference parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='Number of workers for data loading')
    
    # Hardware and execution parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./transformer_model_multi_angle', 
                        help='Directory containing the model checkpoint')
    parser.add_argument('--checkpoint_file', type=str, default='best.pth.tar', 
                        help='Checkpoint file to load (best.pth.tar or last.pth.tar)')
    parser.add_argument('--use_bf16', action='store_true', default=True, 
                        help='Use bfloat16 precision if available')
    
    return parser.parse_args()

def load_model(args):
    """Load the trained model from checkpoint"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    test_dataset = FocusProcessedDataset(base_dir=args.data_dir, split='test', use_multi_angle=args.use_multi_angle)
    logger.info(f"Test dataset size: {len(test_dataset)} samples")
    
    # Get the number of action classes from the dataset
    action_num = len(test_dataset.action_types)
    logger.info(f"Number of action classes: {action_num}")
    
    # Update model constants
    from model import transformer_model
    transformer_model.ENCODER_DIM = args.encoder_dim
    
    # Create model
    model = TransformerActionNet(
        action_num=action_num,
        dropout_rate=args.dropout_rate,
        num_antennas=args.num_antennas,
        data_format='processed'
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    if not os.path.exists(checkpoint_path):
        logger.error(f"No checkpoint found at {checkpoint_path}")
        return None, None
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = model_utils.load_checkpoint(checkpoint_path, model)
    
    # Print model info
    model_size = getModelSize(model)
    logger.info(f"Model parameters: {model_size[1]:,}")
    logger.info(f"Model size: {model_size[-1]:.2f} MB")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    return model, test_dataset, device, checkpoint

def evaluate_model(model, test_dataset, device, args):
    """Evaluate the model on the test dataset"""
    
    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Prepare for metrics
    total_loss = 0
    total_samples = 0
    all_outputs = []
    all_labels = []
    all_predictions = []
    
    # Go through test dataset
    with torch.no_grad():
        with tqdm(total=len(test_dataloader), desc="Testing") as pbar:
            for batch_idx, batch in enumerate(test_dataloader):
                # Move data to device
                if args.use_noisy:
                    RDspecs = batch['noisy_RDspecs'].to(device)
                    AoAspecs = batch['noisy_AoAspecs'].to(device)
                else:
                    RDspecs = batch['sim_RDspecs'].to(device)
                    AoAspecs = batch['sim_AoAspecs'].to(device)
                
                padding_mask = batch['padding_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass with bfloat16 if specified
                if args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output = model(RDspecs, AoAspecs, padding_mask)
                        batch_loss = loss_fn(output, labels)
                else:
                    output = model(RDspecs, AoAspecs, padding_mask)
                    batch_loss = loss_fn(output, labels)
                
                # Get predictions
                _, preds = torch.max(output, 1)
                
                # Collect data for metrics
                total_loss += batch_loss.item() * labels.size(0)
                total_samples += labels.size(0)
                all_outputs.append(output)
                all_labels.append(labels)
                all_predictions.append(preds)
                
                pbar.update(1)
    
    # Compute metrics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Calculate basic metrics
    metrics = classifer_metrics(all_outputs, all_labels)
    avg_loss = total_loss / total_samples
    
    # Get class labels and predictions for detailed analysis
    y_true = all_labels.cpu().numpy()
    y_pred = all_predictions.cpu().numpy()
    
    # Print metrics
    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test Accuracy: {metrics['Action Accuracy']:.4f}")
    logger.info(f"Test F1 Score: {metrics['Action F1 score']:.4f}")
    
    # Get action type names for better visualization
    class_names = test_dataset.action_types
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    logger.info("\nClassification Report:\n" + report)
    
    # Save results to file
    results_path = os.path.join(args.checkpoint_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Test Accuracy: {metrics['Action Accuracy']:.4f}\n")
        f.write(f"Test F1 Score: {metrics['Action F1 score']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Create and save confusion matrix
    create_confusion_matrix(y_true, y_pred, class_names, args.checkpoint_dir)
    
    return metrics, avg_loss

def create_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Create and save a confusion matrix visualization"""
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    logger.info(f"Confusion matrix saved to {save_path}")
    
    # Also save raw counts
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Raw Counts)')
    plt.tight_layout()
    
    # Save raw counts figure
    save_path = os.path.join(save_dir, 'confusion_matrix_raw.png')
    plt.savefig(save_path)
    
    # Save as CSV for further analysis
    np.savetxt(os.path.join(save_dir, 'confusion_matrix.csv'), cm, delimiter=',', fmt='%d')

def analyze_failures(model, test_dataset, device, args):
    """Analyze failure cases to understand model weaknesses"""
    
    # Create dataloader with batch size 1 for easy sample identification
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Track failures
    failures = []
    
    # Go through test dataset
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            # Get sample id
            sample_id = test_dataset.sample_ids[idx]
            
            # Move data to device
            if args.use_noisy:
                RDspecs = batch['noisy_RDspecs'].to(device)
                AoAspecs = batch['noisy_AoAspecs'].to(device)
            else:
                RDspecs = batch['sim_RDspecs'].to(device)
                AoAspecs = batch['sim_AoAspecs'].to(device)
            
            padding_mask = batch['padding_mask'].to(device)
            label = batch['labels'].to(device)
            
            # Forward pass
            output = model(RDspecs, AoAspecs, padding_mask)
            _, pred = torch.max(output, 1)
            
            # Check if prediction is wrong
            if pred.item() != label.item():
                confidence = output[0][pred.item()].item()
                true_class = test_dataset.action_types[label.item()]
                pred_class = test_dataset.action_types[pred.item()]
                
                # Store failure information
                failures.append({
                    'sample_id': sample_id,
                    'true_class': true_class,
                    'pred_class': pred_class,
                    'confidence': confidence
                })
    
    # Analyze failures
    if failures:
        logger.info(f"\nFound {len(failures)} misclassified samples")
        
        # Save failure analysis
        failure_path = os.path.join(args.checkpoint_dir, 'failure_analysis.txt')
        with open(failure_path, 'w') as f:
            f.write(f"Total failures: {len(failures)} out of {len(test_dataset)} test samples\n\n")
            f.write("Top failures by confidence:\n")
            
            # Sort failures by confidence (high to low)
            for failure in sorted(failures, key=lambda x: x['confidence'], reverse=True)[:20]:
                f.write(f"Sample {failure['sample_id']}: True: {failure['true_class']}, " + 
                        f"Predicted: {failure['pred_class']}, Confidence: {failure['confidence']:.4f}\n")
        
        logger.info(f"Failure analysis saved to {failure_path}")
    else:
        logger.info("No misclassifications found!")

def main():
    # Parse arguments
    args = parse_args()
    
    # Load model and dataset
    model, test_dataset, device, checkpoint = load_model(args)
    
    if model is None:
        return
    
    # Get training information from checkpoint
    train_info = ""
    if checkpoint:
        if 'epoch' in checkpoint:
            train_info += f"Epochs trained: {checkpoint['epoch']}\n"
        if 'val_acc' in checkpoint:
            train_info += f"Best validation accuracy: {checkpoint['val_acc']:.4f}\n"
        if 'encoder_dim' in checkpoint:
            train_info += f"Encoder dimension: {checkpoint['encoder_dim']}\n"
        if 'dropout_rate' in checkpoint:
            train_info += f"Dropout rate: {checkpoint['dropout_rate']}\n"
        
        logger.info("Model training information:")
        logger.info(train_info)
    
    # Evaluate model
    logger.info("Starting model evaluation on test set...")
    metrics, loss = evaluate_model(model, test_dataset, device, args)
    
    # Analyze failure cases
    logger.info("Analyzing failure cases...")
    analyze_failures(model, test_dataset, device, args)
    
    logger.info("Testing completed!")
    
if __name__ == "__main__":
    main() 