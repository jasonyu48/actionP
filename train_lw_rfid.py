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
import json
import random

# Import custom modules
from model.net_model_lw import ActionNet, loss_fn, classifer_metrics
import model_utils
from FocusDataset import LWDataset, lw_collate_fn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LW_Focus_Training')

def parse_args():
    parser = argparse.ArgumentParser(description='LW Focus Action Recognition Training')

    parser.add_argument('--checkpoint_dir', type=str, default='./lw_model_rfid_from_scratch', help='Directory to save checkpoints')
    parser.add_argument('--use_bf16', action='store_true', default=True, help='Use bfloat16 precision')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--hyp_search', action='store_true', default=False, help='Perform hyperparameter search before training')
    parser.add_argument('--use_noisy', action='store_true', default=True, help='Use noisy data when available (applies to simulated data only)')
    parser.add_argument('--use_multi_angle', action='store_true', default=True, help='Use data from all angles (0, 90, 180, 270) instead of just 90 degrees')
    
    parser.add_argument('--num_channels', type=int, default=16, help='Number of channels for the model')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--num_epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for model')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA if available')
    parser.add_argument('--use_action', type=str, nargs='+', default=["close","open","pick up","put down","sit down","stand up","wipe"], 
                        help='List of actions to classify')
    
    # New arguments for dataset type and split strategy
    parser.add_argument('--dataset_type', type=str, default='real', choices=['simulated', 'real', 'mixed'], 
                       help='Type of dataset to use (simulated, real, or mixed)')
    parser.add_argument('--split_strategy', type=str, default='random-subset', choices=['default', 'angle-based', 'random-subset'],
                       help='Strategy for splitting data into train/val sets (for non-mixed datasets)')
    parser.add_argument('--data_dir', type=str, default='/weka/scratch/rzhao36/lwang/datasets/HOI/datasets/classic', help='Directory with simulated data')
    parser.add_argument('--real_data_dir', type=str, default='/weka/scratch/rzhao36/lwang/datasets/HOI/RealAction/datasets/classic',
                       help='Directory with real data')
    
    parser.add_argument('--train_angles', type=str, nargs='+', default=['0', '180', '270'], 
                       help='Angles to use for training in angle-based split (for non-mixed datasets)')
    parser.add_argument('--val_angle', type=str, default='90', 
                       help='Angle to use for validation in angle-based split (for non-mixed datasets)')
    parser.add_argument('--samples_per_class', type=int, default=27, 
                       help='Maximum samples per class for random-subset split (for non-mixed datasets)')
    
    # Arguments for finetuning
    parser.add_argument('--finetune', action='store_true', default=False, 
                       help='Finetune a model from an existing checkpoint')
    parser.add_argument('--finetune_checkpoint', type=str, default='./lw_model_pretrain_lw/best.pth.tar', 
                       help='Path to load the model for finetuning (required when --finetune is set)')
    
    # Arguments for mixed dataset
    parser.add_argument('--mixed_sim_split', type=str, default='random-subset', choices=['default', 'angle-based', 'random-subset'],
                       help='Split strategy for simulated data in mixed dataset (only used when dataset_type=mixed)')
    parser.add_argument('--sim_train_angles', type=str, nargs='+', default=['0', '180', '270'], 
                       help='Angles to use for training simulated data in mixed angle-based split (only used when dataset_type=mixed)')
    parser.add_argument('--sim_val_angle', type=str, default='90', 
                       help='Angle to use for validation simulated data in mixed angle-based split (only used when dataset_type=mixed)')
    parser.add_argument('--sim_samples_per_class', type=int, default=4, 
                       help='Maximum samples per class for simulated data in mixed random-subset split (only used when dataset_type=mixed)')
    
    parser.add_argument('--mixed_real_split', type=str, default='random-subset', choices=['default', 'angle-based', 'random-subset'],
                       help='Split strategy for real data in mixed dataset (only used when dataset_type=mixed)')
    parser.add_argument('--real_train_angles', type=str, nargs='+', default=['0', '180', '270'], 
                       help='Angles to use for training real data in mixed angle-based split (only used when dataset_type=mixed)')
    parser.add_argument('--real_val_angle', type=str, default='90', 
                       help='Angle to use for validation real data in mixed angle-based split (only used when dataset_type=mixed)')
    parser.add_argument('--real_samples_per_class', type=int, default=27, 
                       help='Maximum samples per class for real data in mixed random-subset split (only used when dataset_type=mixed)')
    
    parser.add_argument('--val_real_only', action='store_true', default=True,
                       help='Use only real data for validation when using mixed datasets (only used when dataset_type=mixed)')
    parser.add_argument('--resample_sim_data', action='store_true', default=False,
                       help='Resample simulated data across different epochs when using mixed dataset with random-subset strategy')
    
    # HOINet specific parameters
    parser.add_argument('--input_dim', type=int, default=256, 
                       help='Input dimension for the HOINet model')
    parser.add_argument('--obj_num', type=int, default=6, 
                       help='Number of objects for the HOINet model')
    parser.add_argument('--obj_dim', type=int, default=4, 
                       help='Object dimension for the HOINet model')
    parser.add_argument('--obj_branch_size', type=int, default=128, 
                       help='Object branch size for the HOINet model')
    parser.add_argument('--loss_coef', type=float, default=0.5, 
                       help='Weight coefficient for object classification loss')
    
    args = parser.parse_args()
    
    # Validate arguments for consistency
    if args.finetune and args.finetune_checkpoint is None:
        raise ValueError("--finetune_checkpoint must be specified when --finetune is set")
    
    # Check if dataset directories exist
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Simulated data directory does not exist: {args.data_dir}")
    
    if (args.dataset_type == 'real' or args.dataset_type == 'mixed') and not os.path.exists(args.real_data_dir):
        raise ValueError(f"Real data directory does not exist: {args.real_data_dir}")
    
    # Warn about unused arguments
    if args.dataset_type != 'mixed':
        if args.mixed_sim_split != 'default' or args.mixed_real_split != 'default':
            logger.warning(f"--mixed_sim_split and --mixed_real_split are only used when dataset_type=mixed")
        
        if args.sim_train_angles != ['0', '180', '270'] or args.sim_val_angle != '90' or \
           args.real_train_angles != ['0', '180', '270'] or args.real_val_angle != '90' or \
           args.sim_samples_per_class is not None or args.real_samples_per_class is not None:
            logger.warning(f"Mixed dataset parameters (sim_*/real_*) are only used when dataset_type=mixed")
            
        if args.val_real_only:
            logger.warning(f"--val_real_only is only used when dataset_type=mixed")
            
        if args.resample_sim_data:
            logger.warning(f"--resample_sim_data is only used when dataset_type=mixed and mixed_sim_split=random-subset")
    
    if args.dataset_type == 'real' and args.use_noisy:
        logger.warning(f"--use_noisy has no effect when dataset_type=real since real data doesn't have noisy versions")
    
    return args

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
    learning_rates = [1e-6,5e-6,1e-5]
    batch_sizes = [8,4,16]
    
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
    
    # Determine data type
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
            collate_fn=lw_collate_fn,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lw_collate_fn,
            pin_memory=True
        )
        
        # Create model
        model = ActionNet(args)
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
                        specs = batch['noisy_specs'].to(device)
                    else:
                        specs = batch['sim_specs'].to(device)
                    
                    padding_mask = batch['padding_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Forward pass with mixed precision if available
                    if args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            output = model(specs, padding_mask)
                            batch_loss = loss_fn(output, labels)
                    else:
                        output = model(specs, padding_mask)
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
        f.write(f"Data type: {data_type}\n\n")
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
    plt.title(f'Hyperparameter Search Results ({data_type} data)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(search_dir, 'search_results.png'))
    
    return best_lr, best_batch_size
    
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, args):
    """Train the model and evaluate on validation set"""
    
    # Import loss function from net_model_lw_with_rfid
    from model.net_model_lw_with_rfid import loss_fn, classifer_metrics
    
    # Determine data type message
    if args.dataset_type == 'mixed':
        data_type = "mixed (simulated + real)"
    elif args.dataset_type == 'real':
        data_type = "real"
    else:
        data_type = "noisy simulated" if args.use_noisy else "simulated"
    
    logger.info(f"Training using {data_type} data")
    
    # Check for checkpoints directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    # Save parameters to checkpoint directory
    save_args_to_json(args, os.path.join(args.checkpoint_dir, 'params.json'))
    
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
    best_train_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Starting epoch
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume or args.finetune:
        checkpoint_path = args.finetune_checkpoint if args.finetune else os.path.join(args.checkpoint_dir, 'best.pth.tar')
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = model_utils.load_checkpoint(checkpoint_path, model, optimizer)
            
            # If finetuning, don't resume epoch or metrics
            if not args.finetune:
                start_epoch = checkpoint['epoch']
                best_val_acc = checkpoint.get('val_acc', 0.0)
                train_losses = checkpoint.get('train_losses', [])
                val_losses = checkpoint.get('val_losses', [])
                train_accs = checkpoint.get('train_accs', [])
                val_accs = checkpoint.get('val_accs', [])
                logger.info(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
            else:
                logger.info(f"Checkpoint loaded for finetuning. Starting from epoch 0.")
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
        all_action_preds = []
        all_action_labels = []
        all_obj_preds = []
        all_obj_labels = []
        
        # Use tqdm for progress bar
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]") as pbar:
            for batch_idx, batch_data in enumerate(train_dataloader):
                # Handle different batch format for mixed dataset
                if args.dataset_type == 'mixed':
                    batch, src = batch_data  # batch_data is tuple of (batch, source)
                    # Log source type periodically
                    if batch_idx % 20 == 0:
                        logger.info(f"Batch {batch_idx} source: {src}")
                else:
                    batch = batch_data  # batch_data is just the batch
                
                # Prepare mmWave data for model
                if args.use_noisy:
                    specs = batch['noisy_specs'].to(device)
                else:
                    specs = batch['sim_specs'].to(device)
                
                padding_mask = batch['padding_mask'].to(device)
                action_labels = batch['labels'].to(device)
                
                # Prepare RFID data for model
                obj = batch['obj'].to(device)
                obj_mask = batch['obj_mask'].to(device)
                obj_name = batch['obj_name'].to(device)
                obj_labels = batch['obj_labels'].to(device)
                
                # Organize data for HOINet format
                mm_data_list = [specs, padding_mask]
                rfid_data_list = [obj, obj_mask, obj_name]
                
                # Convert to bfloat16 if using mixed precision
                if args.use_bf16 and torch.cuda.get_device_capability()[0] >= 8:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # Forward pass for HOINet
                        action_output, obj_output = model(mm_data_list, rfid_data_list)
                        # Use the loss function from HOINet
                        batch_loss = loss_fn(action_output, action_labels, obj_output, obj_labels, args)
                else:
                    # Forward pass (standard precision)
                    action_output, obj_output = model(mm_data_list, rfid_data_list)
                    batch_loss = loss_fn(action_output, action_labels, obj_output, obj_labels, args)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += batch_loss.item() * action_labels.size(0)
                epoch_samples += action_labels.size(0)
                
                # Track predictions
                _, action_preds = torch.max(action_output, 1)
                _, obj_preds = torch.max(obj_output, 1)
                all_action_preds.extend(action_preds.cpu().numpy())
                all_action_labels.extend(action_labels.cpu().numpy())
                all_obj_preds.extend(obj_preds.cpu().numpy())
                all_obj_labels.extend(obj_labels.cpu().numpy())
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': batch_loss.item()})
        
        # Calculate epoch metrics
        train_loss = epoch_loss / epoch_samples
        action_acc = np.mean(np.array(all_action_preds) == np.array(all_action_labels))
        obj_acc = np.mean(np.array(all_obj_preds) == np.array(all_obj_labels))
        
        # Evaluate on validation set
        val_loss, val_metrics = evaluate(model, val_dataloader, device, args)
        val_acc = val_metrics['Action Accuracy']
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Action Acc: {action_acc:.4f}, Train Obj Acc: {obj_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Action Acc: {val_acc:.4f}, Val Obj Acc: {val_metrics['Object Accuracy']:.4f}")
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(action_acc)  # We use action accuracy as the primary metric
        val_accs.append(val_acc)
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_train_acc = action_acc
            best_epoch = epoch + 1
            
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
            'use_multi_angle': args.use_multi_angle,
            'dataset_type': args.dataset_type,
            'split_strategy': args.split_strategy
        }
        
        model_utils.save_checkpoint(state, is_best, args.checkpoint_dir)
        
        # Save training plots
        save_plot(train_losses, val_losses, train_accs, val_accs, args)
    
    # Save best accuracy information
    best_metrics = {
        'best_val_acc': best_val_acc, # action accuracy
        'best_train_acc': best_train_acc,
        'best_epoch': best_epoch,
        'dataset_type': args.dataset_type,
        'split_strategy': args.split_strategy
    }
    
    # Save best metrics to file
    best_metrics_path = os.path.join(args.checkpoint_dir, 'best_metrics.json')
    with open(best_metrics_path, 'w') as f:
        json.dump(best_metrics, f, indent=4)
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}, with training accuracy: {best_train_acc:.4f}")
    logger.info(f"Best metrics saved to {best_metrics_path}")

def evaluate(model, dataloader, device, args):
    """Evaluate the model on the given dataloader"""
    model.eval()
    
    # Import evaluation metrics
    from model.net_model_lw_with_rfid import loss_fn, classifer_metrics
    
    total_loss = 0
    total_samples = 0
    all_action_outputs = []
    all_action_labels = []
    all_obj_outputs = []
    all_obj_labels = []
    
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluation") as pbar:
            for batch_idx, batch_data in enumerate(dataloader):
                # Handle different batch format for mixed dataset
                if args.dataset_type == 'mixed' and not args.val_real_only:
                    batch, _ = batch_data  # Ignore source for evaluation
                else:
                    batch = batch_data
                
                # Prepare mmWave data for model
                if args.use_noisy:
                    specs = batch['noisy_specs'].to(device)
                else:
                    specs = batch['sim_specs'].to(device)
                
                padding_mask = batch['padding_mask'].to(device)
                action_labels = batch['labels'].to(device)
                
                # Prepare RFID data for model
                obj = batch['obj'].to(device)
                obj_mask = batch['obj_mask'].to(device)
                obj_name = batch['obj_name'].to(device)
                obj_labels = batch['obj_labels'].to(device)
                
                # Organize data for HOINet format
                mm_data_list = [specs, padding_mask]
                rfid_data_list = [obj, obj_mask, obj_name]
                
                # Forward pass with bfloat16 if specified
                if args.use_bf16 and torch.cuda.get_device_capability()[0] >= 8:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        action_output, obj_output = model(mm_data_list, rfid_data_list)
                        batch_loss = loss_fn(action_output, action_labels, obj_output, obj_labels, args)
                else:
                    action_output, obj_output = model(mm_data_list, rfid_data_list)
                    batch_loss = loss_fn(action_output, action_labels, obj_output, obj_labels, args)
                
                # Update metrics
                total_loss += batch_loss.item() * action_labels.size(0)
                total_samples += action_labels.size(0)
                
                # Track outputs for metrics
                all_action_outputs.append(action_output)
                all_action_labels.append(action_labels)
                all_obj_outputs.append(obj_output)
                all_obj_labels.append(obj_labels)
                
                pbar.update(1)
    
    # Calculate metrics
    all_action_outputs = torch.cat(all_action_outputs, dim=0)
    all_action_labels = torch.cat(all_action_labels, dim=0)
    all_obj_outputs = torch.cat(all_obj_outputs, dim=0)
    all_obj_labels = torch.cat(all_obj_labels, dim=0)
    
    metrics = classifer_metrics(all_action_outputs, all_action_labels, all_obj_outputs, all_obj_labels)
    
    avg_loss = total_loss / total_samples
    
    return avg_loss, metrics

def save_args_to_json(args, json_path):
    """Save command-line arguments to JSON file"""
    args_dict = vars(args)
    with open(json_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

class MixedDataLoader:
    """Efficient dataloader that interleaves batches from a simulated and a real dataset
    without ever materialising (and discarding) unused batches.  At every epoch we
    build a shuffled list containing the desired *order* of dataset sources
    ("sim" / "real") whose length equals the number of batches in each individual
    DataLoader.  Then we just draw the next batch from the corresponding iterator.

    This removes the previous implementation's costly "skip_count" loops that
    loaded and pinned large batches only to throw them away, a major source of
    host-RAM bloat and I/O overhead.
    """

    def __init__(
        self,
        sim_dataset,
        real_dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 1,
        collate_fn=None,
        drop_last: bool = False,
        pin_memory: bool = True,
    ):
        # Build the underlying DataLoaders
        self.sim_loader = DataLoader(
            sim_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        self.real_loader = DataLoader(
            real_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        self.shuffle = shuffle

    # ------------------------------------------------------------------
    # Python iterator protocol
    # ------------------------------------------------------------------
    def __iter__(self):
        # Fresh iterator for each epoch
        self.sim_iter = iter(self.sim_loader)
        self.real_iter = iter(self.real_loader)

        # Build the epoch sequence of sources ("sim" / "real")
        self.batch_sequence = ["sim"] * len(self.sim_loader) + [
            "real"
        ] * len(self.real_loader)
        if self.shuffle:
            random.shuffle(self.batch_sequence)
        self._seq_idx = 0
        return self

    def __next__(self):
        while self._seq_idx < len(self.batch_sequence):
            src = self.batch_sequence[self._seq_idx]
            self._seq_idx += 1

            if src == "sim":
                try:
                    batch = next(self.sim_iter)
                    return batch, src
                except StopIteration:
                    # simulated loader exhausted early → fall through and continue
                    continue
            else:  # src == "real"
                try:
                    batch = next(self.real_iter)
                    return batch, src
                except StopIteration:
                    continue
        # If we reach here, both loaders are exhausted
        raise StopIteration

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    def __len__(self):
        # Number of batches produced in one epoch
        return len(self.sim_loader) + len(self.real_loader)

class DynamicMixedDataLoader(MixedDataLoader):
    """Extended version of MixedDataLoader that recreates the simulated dataset
    with a fresh random subset before each epoch (when using random-subset strategy).
    This allows different samples to be used for each epoch during training.
    
    Only the simulated dataset is resampled; the real dataset remains the same.
    """

    def __init__(
        self,
        sim_dataset,
        real_dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 1,
        collate_fn=None,
        drop_last: bool = False,
        pin_memory: bool = True,
        # New parameters for creating new sim dataset each epoch
        data_dir=None,
        use_multi_angle=True,
        split_strategy=None,
        train_angles=None,
        val_angle=None,
        samples_per_class=None,
        resample_sim_data=True,
    ):
        # Store the original mixed dataset configuration
        self.data_dir = data_dir
        self.use_multi_angle = use_multi_angle
        self.split_strategy = split_strategy
        self.train_angles = train_angles
        self.val_angle = val_angle
        self.samples_per_class = samples_per_class
        self.resample_sim_data = resample_sim_data
        
        # Store other dataloader parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        
        # Store the real dataset (never changes)
        self.real_dataset = real_dataset
        
        # Only store a reference to the initial sim dataset, not used after first epoch
        self.initial_sim_dataset = sim_dataset
        
        # Create the initial dataloader for real dataset
        self.real_loader = DataLoader(
            real_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        
        # Initialize with the first sim dataset
        self._create_sim_loader(sim_dataset)
    
    def _create_sim_loader(self, sim_dataset):
        """Create a new dataloader for the simulated dataset"""
        self.sim_loader = DataLoader(
            sim_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        
    def _create_new_sim_dataset(self):
        """Create a new simulated dataset with a different random subset"""
        # Generate a new random seed for this dataset creation
        new_seed = int(time.time()) + random.randint(1, 10000)
        logger.info(f"Creating new simulated dataset with different random subset using seed {new_seed}")
        
        # Create a new simulated dataset with the same configuration but a new seed
        new_sim_dataset = LWDataset(
            base_dir=self.data_dir,
            split='train',
            use_multi_angle=self.use_multi_angle,
            use_real_data=False,
            split_strategy=self.split_strategy,
            train_angles=self.train_angles,
            val_angle=self.val_angle,
            samples_per_class=self.samples_per_class,
            seed=new_seed  # Use the new random seed
        )
        
        # Create a new dataloader with the new dataset
        self._create_sim_loader(new_sim_dataset)
        
    def __iter__(self):
        # Only recreate the simulated dataset if using random-subset strategy and resampling is enabled
        if self.split_strategy == 'random-subset' and self.resample_sim_data:
            self._create_new_sim_dataset()
        
        # Fresh iterators for each epoch
        self.sim_iter = iter(self.sim_loader)
        self.real_iter = iter(self.real_loader)

        # Build the batch sequence
        self.batch_sequence = ["sim"] * len(self.sim_loader) + [
            "real"
        ] * len(self.real_loader)
        if self.shuffle:
            random.shuffle(self.batch_sequence)
        self._seq_idx = 0
        return self

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    model_utils.setup_seed(args.seed)
    
    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Validate finetune checkpoint if provided
    if args.finetune:
        if not os.path.exists(args.finetune_checkpoint):
            raise ValueError(f"Finetune checkpoint does not exist: {args.finetune_checkpoint}")
        logger.info(f"Will finetune from checkpoint: {args.finetune_checkpoint}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Save arguments to JSON file
    args_json_path = os.path.join(args.checkpoint_dir, 'args.json')
    save_args_to_json(args, args_json_path)
    logger.info(f"Saved arguments to {args_json_path}")
    
    # Note: HOINet parameters are now set via command line arguments in parse_args()
    
    # Log dataset configuration
    logger.info(f"Dataset type: {args.dataset_type}")

    # Import HOINet model instead of ActionNet
    from model.net_model_lw_with_rfid import HOINet, loss_fn, classifer_metrics

    # Import the modified dataloader
    from new_data_loader import fetch_dataloader, lw_collate_fn, MixedDataLoader
    
    # Create datasets based on configuration
    if args.dataset_type == 'mixed':
        # Create dataloaders with the modified fetch_dataloader
        dataloaders = fetch_dataloader(['train', 'val'], args.real_data_dir, args.data_dir, args)
        train_dataloader = dataloaders['train']
        val_dataloader = dataloaders['val']
    else:
        # Use the dataloader directly from new_data_loader
        dataloaders = fetch_dataloader(['train', 'val'], args.real_data_dir, args.data_dir, args)
        train_dataloader = dataloaders['train']
        val_dataloader = dataloaders['val']
    
    # Create HOINet model instead of ActionNet
    model = HOINet(args)
    model = model.to(device)
    
    # Print model summary
    logger.info(f"Model size: {model_utils.getModelSize(model)[-1]:.2f} MB")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1)
    
    # Record training configuration
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    with open(os.path.join(args.checkpoint_dir, 'training_config.txt'), 'w') as f:
        f.write(f"Dataset type: {args.dataset_type}\n")
        if args.dataset_type == 'mixed':
            f.write(f"Sim split strategy: {args.mixed_sim_split}\n")
            f.write(f"Real split strategy: {args.mixed_real_split}\n")
            f.write(f"Validation using real data only: {args.val_real_only}\n")
            
            # Add the separate parameters for mixed datasets
            if args.mixed_sim_split == 'angle-based':
                f.write(f"Sim train angles: {args.sim_train_angles}\n")
                f.write(f"Sim val angle: {args.sim_val_angle}\n")
            if args.mixed_sim_split == 'random-subset':
                f.write(f"Sim samples per class: {args.sim_samples_per_class}\n")
                
            if args.mixed_real_split == 'angle-based':
                f.write(f"Real train angles: {args.real_train_angles}\n")
                f.write(f"Real val angle: {args.real_val_angle}\n")
            if args.mixed_real_split == 'random-subset':
                f.write(f"Real samples per class: {args.real_samples_per_class}\n")
        else:
            # For non-mixed datasets, log the original parameters
            f.write(f"Split strategy: {args.split_strategy}\n")
            if args.split_strategy == 'angle-based':
                f.write(f"Train angles: {args.train_angles}\n")
                f.write(f"Val angle: {args.val_angle}\n")
            if args.split_strategy == 'random-subset':
                f.write(f"Samples per class: {args.samples_per_class}\n")
            
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"BFloat16: {args.use_bf16}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of epochs: {args.num_epochs}\n")
        f.write(f"Use noisy data: {args.use_noisy}\n")
        f.write(f"Using multi-angle data: {args.use_multi_angle}\n")
        if args.finetune:
            f.write(f"Finetuning from: {args.finetune_checkpoint}\n")
    
    # Train and evaluate
    train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, scheduler, args)
    
if __name__ == "__main__":
    main() 