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
from model.network_cube_learn import RDAT_3DCNNLSTM
from FocusDataset import CubeLearnDataset, cubelearn_collate_fn
import model_utils

# NOTE: This script works with radar data loaded as complex64 data type
# The RDAT_3DCNNLSTM model is designed to handle complex input data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CubeLearn_Training')

def parse_args():
    parser = argparse.ArgumentParser(description='CubeLearn Action Recognition Training')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='/scratch/tshu2/jyu197/XRF55-repo/Focus_processed_multi_angle', 
                        help='Directory of data')
    parser.add_argument('--use_multi_angle', action='store_true', default=True, 
                        help='Use data from all angles (0, 90, 180, 270) instead of just 90 degrees')
    parser.add_argument('--use_noisy', action='store_true', default=True, 
                        help='Use noisy data for training instead of simulated data')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=60, 
                        help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='Number of workers for data loading')
    
    # Hardware and execution parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./cubelearn_model_noise_multiangle', 
                        help='Directory to save checkpoints')
    parser.add_argument('--use_bf16', action='store_true', default=False, 
                        help='Use bfloat16 precision (Note: Not compatible with complex64 data)')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true', default=True, 
                        help='Use CUDA for training if available')
    parser.add_argument('--hyp_search', action='store_true', default=False, 
                        help='Perform hyperparameter search before training')
    
    return parser.parse_args()

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
    
    metrics = {
        'Action Accuracy': action_acc,
        'Action F1 score': action_f1,
    }     
    return metrics

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
    learning_rates = [1e-4, 1e-3, 1e-5, 1e-2]
    batch_sizes = [8]
    
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
    logger.info(f"Hyperparameter search using CubeLearn dataset with {data_type} data")
    
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
            collate_fn=cubelearn_collate_fn,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=cubelearn_collate_fn,
            pin_memory=True
        )
        
        # Create model
        model = RDAT_3DCNNLSTM()
        model = model.to(device)
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        
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
                        radar_frames = batch['noisy_radar_frames'].to(device)
                    else:
                        radar_frames = batch['sim_radar_frames'].to(device)
                    
                    padding_mask = batch['padding_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass - bfloat16 is disabled for complex data
                    output = model(radar_frames)
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
        f.write(f"Dataset: CubeLearn, Data type: {data_type}\n\n")
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
        plt.plot([r['lr'] for r in bs_results], [r['val_acc'] for r in bs_results], marker='o', label=f'BS={bs}')
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Hyperparameter Search Results (CubeLearn dataset, {data_type} data)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(search_dir, 'search_results.png'))
    
    return best_lr, best_batch_size

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

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, args):
    """Train the model and evaluate on validation set"""
    
    # Determine data type
    data_type = "noisy" if args.use_noisy else "simulated"
    logger.info(f"Training using {data_type} data")
    
    # Check for checkpoints directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    # Setup for mixed precision training with bfloat16
    if args.use_bf16:
        logger.warning("BFloat16 is not compatible with complex64 data. Disabling bfloat16 mixed precision.")
        args.use_bf16 = False
        logger.info("Using full precision for complex data.")
    else:
        logger.info("Using full precision for complex data.")
    
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
                    radar_frames = batch['noisy_radar_frames'].to(device)
                else:
                    radar_frames = batch['sim_radar_frames'].to(device)
                
                padding_mask = batch['padding_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass (using full precision for complex data)
                output = model(radar_frames)
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
            'use_noisy': args.use_noisy,
            'use_multi_angle': args.use_multi_angle
        }
        
        model_utils.save_checkpoint(state, is_best, args.checkpoint_dir)
        
        # Save training plots
        save_plot(train_losses, val_losses, train_accs, val_accs, args)
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    global BEST_VAL_ACC
    BEST_VAL_ACC = best_val_acc

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
                    radar_frames = batch['noisy_radar_frames'].to(device)
                else:
                    radar_frames = batch['sim_radar_frames'].to(device)
                
                padding_mask = batch['padding_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass (using full precision for complex data)
                output = model(radar_frames)
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
    
    # Log angle choice
    if args.use_multi_angle:
        logger.info("Using data from all angles (0, 90, 180, 270)")
    else:
        logger.info("Using data from only 90-degree angle")
    
    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Log data type
    logger.info("Loading radar data as complex64 data type")
    
    # Check for mixed precision and complex data compatibility
    if args.use_bf16:
        logger.warning("BFloat16 mixed precision is incompatible with complex64 data and will be disabled")
        args.use_bf16 = False
    
    # Create datasets
    train_dataset = CubeLearnDataset(base_dir=args.data_dir, split='train', use_multi_angle=args.use_multi_angle)
    val_dataset = CubeLearnDataset(base_dir=args.data_dir, split='val', use_multi_angle=args.use_multi_angle)
    
    # Perform hyperparameter search if specified
    if args.hyp_search:
        logger.info("Starting hyperparameter search")
        best_lr, best_batch_size = hyperparameter_search(train_dataset, val_dataset, device, args)
        
        # Update arguments with best values
        args.learning_rate = best_lr
        args.batch_size = best_batch_size
        logger.info(f"Using best hyperparameters: LR={best_lr}, BS={best_batch_size}")
    else:
        logger.info(f"Using default hyperparameters: LR={args.learning_rate}, BS={args.batch_size}")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cubelearn_collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cubelearn_collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = RDAT_3DCNNLSTM()
    model = model.to(device)
    
    # Print model summary
    logger.info(f"Model size: {model_utils.getModelSize(model)[-1]:.2f} MB")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.5)
    
    # Record training configuration
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    with open(os.path.join(args.checkpoint_dir, 'training_config.txt'), 'w') as f:
        f.write(f"Data type: {args.use_noisy and 'noisy' or 'simulated'} (complex64)\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"BFloat16: False (disabled - incompatible with complex64 data)\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of epochs: {args.num_epochs}\n")
        f.write(f"Using multi-angle data: {args.use_multi_angle}\n")
        f.write(f"Hyperparameter search used: {args.hyp_search}\n")
        f.write(f"Model size: {model_utils.getModelSize(model)[-1]:.2f} MB\n")
    
    # Train and evaluate
    train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, args)

    with open(os.path.join(args.checkpoint_dir, 'training_config.txt'), 'a') as f:
        f.write(f"Best validation accuracy: {BEST_VAL_ACC:.4f}\n")
    
if __name__ == "__main__":
    main() 