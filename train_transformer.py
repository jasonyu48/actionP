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
import itertools

# Import custom modules
from model.transformer_model import TransformerActionNet, loss_fn, classifer_metrics, getModelSize
from FocusDataset import FocusProcessedDataset, collate_fn
import model_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Transformer_Training')

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Action Recognition Training')
    
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
                        help='Use noisy data for training instead of simulated data')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=60, 
                        help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='Number of workers for data loading')
    
    # Hardware and execution parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./transformer_model_multi_angle_small', 
                        help='Directory to save checkpoints')
    parser.add_argument('--use_bf16', action='store_true', default=True, 
                        help='Use bfloat16 precision if available')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--hyp_search', action='store_true', default=False, 
                        help='Perform hyperparameter search before training')
    
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

def hyperparameter_search(train_dataset, val_dataset, device, args):
    """
    Perform hyperparameter search to find optimal learning rate and batch size
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: Device to run on
        args: Command line arguments
        
    Returns:
        best_lr: Best learning rate
        best_batch_size: Best batch size
    """
    # Hyperparameter search space
    learning_rates = [5e-5, 5e-4, 1e-4]
    batch_sizes = [32]
    
    # Create results directory for the search
    search_dir = os.path.join(args.checkpoint_dir, 'hyperparam_search')
    if not os.path.exists(search_dir):
        os.makedirs(search_dir)
        
    # Initialize best values
    best_val_acc = 0.0
    best_lr = None
    best_batch_size = None
    
    # Number of epochs for each configuration
    search_epochs = 3
    
    # Store results for plotting
    results = []
    
    # Determine data type based on arguments
    data_type = "noisy" if args.use_noisy else "sim"
    logger.info(f"Hyperparameter search using {data_type} data")
    
    # Try all combinations
    combos = list(itertools.product(learning_rates, batch_sizes))
    logger.info(f"Starting hyperparameter search with {len(combos)} combinations")
    logger.info(f"Search space: LR={learning_rates}, BS={batch_sizes}")
    
    for i, (lr, bs) in enumerate(combos):
        logger.info(f"Testing combination {i+1}/{len(combos)}: LR={lr}, BS={bs}")
        
        # Create dataloaders with current batch size
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=bs, 
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # Get the number of action classes from the dataset
        action_num = len(train_dataset.action_types)
        
        # Create model with current configuration
        model = TransformerActionNet(
            action_num=action_num,
            dropout_rate=args.dropout_rate,
            num_antennas=args.num_antennas,
            data_format='processed'
        )
        model = model.to(device)
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        
        # Training loop for search epochs
        val_accs = []
        
        for epoch in range(search_epochs):
            # Train one epoch
            model.train()
            epoch_loss = 0
            epoch_samples = 0
            
            with tqdm(total=len(train_dataloader), desc=f"Search [{i+1}/{len(combos)}] Epoch {epoch+1}/{search_epochs}") as pbar:
                for batch_idx, batch in enumerate(train_dataloader):
                    # Move data to device based on data type selected
                    if args.use_noisy:
                        RDspecs = batch['noisy_RDspecs'].to(device)
                        AoAspecs = batch['noisy_AoAspecs'].to(device)
                    else:
                        RDspecs = batch['sim_RDspecs'].to(device)
                        AoAspecs = batch['sim_AoAspecs'].to(device)
                    
                    padding_mask = batch['padding_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass with mixed precision if available
                    if args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            output = model(RDspecs, AoAspecs, padding_mask)
                            batch_loss = loss_fn(output, labels)
                    else:
                        output = model(RDspecs, AoAspecs, padding_mask)
                        batch_loss = loss_fn(output, labels)
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    epoch_loss += batch_loss.item() * labels.size(0)
                    epoch_samples += labels.size(0)
                    
                    pbar.update(1)
                    pbar.set_postfix({'loss': batch_loss.item()})
            
            # Evaluate on validation set
            val_loss, val_metrics = evaluate(model, val_dataloader, device, args)
            val_acc = val_metrics['Action Accuracy']
            val_accs.append(val_acc)
            
            logger.info(f"Search [{i+1}/{len(combos)}] Epoch {epoch+1}/{search_epochs} - "
                       f"Train Loss: {epoch_loss/epoch_samples:.4f}, Val Acc: {val_acc:.4f}")
        
        # Calculate final validation accuracy for this configuration (average of last 2 epochs if available)
        if len(val_accs) > 1:
            final_val_acc = sum(val_accs[-2:]) / 2
        else:
            final_val_acc = val_accs[-1]
            
        # Store result
        results.append({
            'lr': lr,
            'batch_size': bs,
            'val_acc': final_val_acc
        })
        
        # Update best configuration if needed
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_lr = lr
            best_batch_size = bs
            
        # Clear memory
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Print and save results
    logger.info("\nHyperparameter Search Results:")
    results_file = os.path.join(search_dir, 'search_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Dataset type: processed, Data type: {data_type}\n\n")
        for result in sorted(results, key=lambda x: x['val_acc'], reverse=True):
            result_str = f"LR: {result['lr']}, BS: {result['batch_size']}, Val Acc: {result['val_acc']:.4f}"
            logger.info(result_str)
            f.write(result_str + '\n')
    
    logger.info(f"\nBest Configuration: LR={best_lr}, BS={best_batch_size}, Val Acc={best_val_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for bs in batch_sizes:
        bs_results = [r for r in results if r['batch_size'] == bs]
        bs_results = sorted(bs_results, key=lambda x: x['lr'])
        if bs_results:  # Only plot if there are results for this batch size
            plt.plot([r['lr'] for r in bs_results], [r['val_acc'] for r in bs_results], marker='o', label=f'BS={bs}')
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Hyperparameter Search Results (processed dataset, {data_type} data)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(search_dir, 'search_results.png'))
    
    return best_lr, best_batch_size
    
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, args):
    """Train the model and evaluate on validation set"""
    
    # Determine data type
    data_type = "noisy" if args.use_noisy else "simulated"
    logger.info(f"Training using processed dataset with {data_type} data")
    
    # Check for checkpoints directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
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
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        
        # Set model to training mode
        model.train()
        
        # Track metrics
        epoch_loss = 0
        epoch_samples = 0
        all_preds = []
        all_labels = []
        
        # Use tqdm for progress bar
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]") as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                # Move data to device based on data type selected
                if args.use_noisy:
                    RDspecs = batch['noisy_RDspecs'].to(device)
                    AoAspecs = batch['noisy_AoAspecs'].to(device)
                else:
                    RDspecs = batch['sim_RDspecs'].to(device)
                    AoAspecs = batch['sim_AoAspecs'].to(device)
                
                padding_mask = batch['padding_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Convert to bfloat16 if using mixed precision
                if args.use_bf16 and torch.cuda.get_device_capability()[0] >= 8:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # Forward pass
                        output = model(RDspecs, AoAspecs, padding_mask)
                        batch_loss = loss_fn(output, labels)
                else:
                    # Forward pass (standard precision)
                    output = model(RDspecs, AoAspecs, padding_mask)
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
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - "
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
            'val_acc': val_acc,
            'encoder_dim': args.encoder_dim,
            'use_noisy': args.use_noisy,
            'use_multi_angle': args.use_multi_angle,
            'dropout_rate': args.dropout_rate,
            'num_antennas': args.num_antennas
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
                # Move data to device based on data type selected
                if args.use_noisy:
                    RDspecs = batch['noisy_RDspecs'].to(device)
                    AoAspecs = batch['noisy_AoAspecs'].to(device)
                else:
                    RDspecs = batch['sim_RDspecs'].to(device)
                    AoAspecs = batch['sim_AoAspecs'].to(device)
                
                padding_mask = batch['padding_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass with bfloat16 if specified
                if args.use_bf16 and torch.cuda.get_device_capability()[0] >= 8:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output = model(RDspecs, AoAspecs, padding_mask)
                        batch_loss = loss_fn(output, labels)
                else:
                    output = model(RDspecs, AoAspecs, padding_mask)
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
    
    # Update model constants
    # Note: This requires importing and modifying the transformer_model module
    from model import transformer_model
    transformer_model.ENCODER_DIM = args.encoder_dim
    
    # Log model settings
    logger.info(f"Model settings: ENCODER_DIM={args.encoder_dim}, dropout={args.dropout_rate}, antennas={args.num_antennas}")
    
    # Log data type choice
    if args.use_noisy:
        logger.info("Using noisy data for training")
    else:
        logger.info("Using simulated data for training")
    
    # Log angle choice
    if args.use_multi_angle:
        logger.info("Using data from all angles (0, 90, 180, 270)")
    else:
        logger.info("Using data from only 90-degree angle")
    
    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = FocusProcessedDataset(base_dir=args.data_dir, split='train', use_multi_angle=args.use_multi_angle)
    val_dataset = FocusProcessedDataset(base_dir=args.data_dir, split='val', use_multi_angle=args.use_multi_angle)
    
    # Perform hyperparameter search if specified
    if args.hyp_search:
        logger.info("Starting hyperparameter search")
        best_lr, best_batch_size = hyperparameter_search(train_dataset, val_dataset, device, args)
        
        # Update parameters with best values
        args.learning_rate = best_lr
        args.batch_size = best_batch_size
        logger.info(f"Using best hyperparameters: LR={best_lr}, BS={best_batch_size}")
    else:
        logger.info(f"Using default hyperparameters: LR={args.learning_rate}, BS={args.batch_size}")
    
    # Create dataloaders with selected batch size
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Get the number of action classes from the dataset
    action_num = len(train_dataset.action_types)
    logger.info(f"Number of action classes: {action_num}")
    
    # Create model
    model = TransformerActionNet(
        action_num=action_num,
        dropout_rate=args.dropout_rate,
        num_antennas=args.num_antennas,
        data_format='processed'
    )
    model = model.to(device)
    
    # Print model summary
    model_size = getModelSize(model)
    logger.info(f"Model parameters: {model_size[1]:,}")
    logger.info(f"Model size: {model_size[-1]:.2f} MB")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1)
    
    # Record training configuration
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    with open(os.path.join(args.checkpoint_dir, 'training_config.txt'), 'w') as f:
        f.write(f"Dataset: processed\n")
        f.write(f"Data type: {args.use_noisy and 'noisy' or 'simulated'}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"BFloat16: {args.use_bf16}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of epochs: {args.num_epochs}\n")
        f.write(f"Using multi-angle data: {args.use_multi_angle}\n")
        f.write(f"Encoder dimension: {args.encoder_dim}\n")
        f.write(f"Dropout rate: {args.dropout_rate}\n")
        f.write(f"Number of antennas: {args.num_antennas}\n")
        f.write(f"Model size: {model_size[-1]:.2f} MB\n")
        f.write(f"Model parameters: {model_size[1]:,}\n")
    
    # Train and evaluate
    train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, args)
    
if __name__ == "__main__":
    main() 