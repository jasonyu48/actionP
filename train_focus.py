import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import logging
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
from model.net_model import ActionNet, loss_fn, classifer_metrics
import model_utils
from FocusDataset import FocusProcessedDataset, collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Focus_Training')

def parse_args():
    parser = argparse.ArgumentParser(description='Focus Action Recognition Training')
    parser.add_argument('--params_path', type=str, default='params.json', help='Path to parameters JSON file')
    parser.add_argument('--data_dir', type=str, default='./Focus_processed', help='Directory with processed data')
    parser.add_argument('--checkpoint_dir', type=str, default='./focus_model', help='Directory to save checkpoints')
    parser.add_argument('--use_bf16', action='store_true', default=True, help='Use bfloat16 precision')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

def save_plot(train_losses, val_losses, train_accs, val_accs, args):
    """Save training and validation metrics plots"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_metrics.png'))
    
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, args, params):
    """Train the model and evaluate on validation set"""
    
    # Check for checkpoints directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    # Save parameters to checkpoint directory
    params.save(os.path.join(args.checkpoint_dir, 'params.json'))
    
    # Setup for mixed precision training with bfloat16
    scaler = None
    if args.use_bf16 and torch.cuda.is_available():
        # Get device capability
        device_capability = torch.cuda.get_device_capability()
        has_bf16_support = device_capability[0] >= 8  # Ampere (sm_80) and above support BF16
        
        if has_bf16_support:
            logger.info("Using bfloat16 mixed precision training")
            # No scaler is needed for BF16, unlike FP16
        else:
            logger.info("BF16 not supported on this device. Using FP32.")
            args.use_bf16 = False
    else:
        logger.info("Not using mixed precision training")
    
    # Variables to track best model and training history
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Starting epoch
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best.pth.tar')
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = model_utils.load_checkpoint(checkpoint_path, model, optimizer)
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint.get('val_acc', 0.0)
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            train_accs = checkpoint.get('train_accs', [])
            val_accs = checkpoint.get('val_accs', [])
            logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    
    # Training loop
    for epoch in range(start_epoch, params.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{params.num_epochs}")
        
        # Set model to training mode
        model.train()
        
        # Track metrics
        epoch_loss = 0
        epoch_samples = 0
        all_preds = []
        all_labels = []
        
        # Use tqdm for progress bar
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{params.num_epochs} [Train]") as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                # Move data to device
                sim_RDspecs = batch['sim_RDspecs'].to(device)
                sim_AoAspecs = batch['sim_AoAspecs'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Convert to bfloat16 if using mixed precision
                if args.use_bf16 and has_bf16_support:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        # Forward pass
                        output = model(sim_RDspecs, sim_AoAspecs, padding_mask)
                        batch_loss = loss_fn(output, labels)
                else:
                    # Forward pass (standard precision)
                    output = model(sim_RDspecs, sim_AoAspecs, padding_mask)
                    batch_loss = loss_fn(output, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += batch_loss.item() * labels.size(0)
                epoch_samples += labels.size(0)
                
                # Track predictions
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': batch_loss.item()})
        
        # Calculate epoch metrics
        train_loss = epoch_loss / epoch_samples
        train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Evaluate on validation set
        val_loss, val_metrics = evaluate(model, val_dataloader, device, args)
        val_acc = val_metrics['Action Accuracy']
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{params.num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            
        # Checkpoint state
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'val_acc': val_acc
        }
        
        model_utils.save_checkpoint(state, is_best, args.checkpoint_dir)
        
        # Save training plots
        save_plot(train_losses, val_losses, train_accs, val_accs, args)
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")

def evaluate(model, dataloader, device, args):
    """Evaluate the model on the given dataloader"""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluation") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                # Move data to device
                sim_RDspecs = batch['sim_RDspecs'].to(device)
                sim_AoAspecs = batch['sim_AoAspecs'].to(device)
                padding_mask = batch['padding_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass with bfloat16 if specified
                if args.use_bf16 and torch.cuda.get_device_capability()[0] >= 8:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        output = model(sim_RDspecs, sim_AoAspecs, padding_mask)
                        batch_loss = loss_fn(output, labels)
                else:
                    output = model(sim_RDspecs, sim_AoAspecs, padding_mask)
                    batch_loss = loss_fn(output, labels)
                
                # Update metrics
                total_loss += batch_loss.item() * labels.size(0)
                total_samples += labels.size(0)
                
                # Track outputs for metrics
                all_outputs.append(output)
                all_labels.append(labels)
                
                pbar.update(1)
    
    # Calculate metrics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = classifer_metrics(all_outputs, all_labels)
    
    avg_loss = total_loss / total_samples
    
    return avg_loss, metrics

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    model_utils.setup_seed(args.seed)
    
    # Load parameters
    params = model_utils.Params(args.params_path)
    
    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() and params.cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = FocusProcessedDataset(base_dir=args.data_dir, split='train')
    val_dataset = FocusProcessedDataset(base_dir=args.data_dir, split='val')
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=params.batch_size, 
        shuffle=True,
        num_workers=params.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = ActionNet(params)
    model = model.to(device)
    
    # Print model summary
    logger.info(f"Model size: {model_utils.getModelSize(model)[-1]:.2f} MB")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5)
    
    # Train and evaluate
    train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, args, params)
    
if __name__ == "__main__":
    main() 