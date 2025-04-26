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
from model.transformer_model_with_rfid import HOINet, loss_fn, classifer_metrics, getModelSize
from FocusDataset import FocusDatasetwithRFID, rfid_collate_fn, FocusRealDatasetwithRFID
import model_utils

BEST_ACTION_ACC = 0.0
BEST_OBJ_ACC = 0.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HOINet_Training')

def parse_args():
    parser = argparse.ArgumentParser(description='HOI Recognition Training with Radar and RFID Data')
    
    # Model parameters
    parser.add_argument('--encoder_dim', type=int, default=256, 
                        help='Dimension of encoder models')
    parser.add_argument('--fusion_dim', type=int, default=512, 
                        help='Dimension for fusion transformer')
    parser.add_argument('--neuron_num', type=int, default=64, 
                        help='Dimension for object branch')
    parser.add_argument('--num_antennas', type=int, default=12, 
                        help='Number of antennas in radar data')
    parser.add_argument('--dropout_rate', type=float, default=0.2, 
                        help='Dropout rate for all models')
    parser.add_argument('--loss_coef', type=float, default=0.3,
                        help='Weight of the object classification loss')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='/scratch/tshu2/jyu197/Focus_processed_multi_angle_rfid', 
                        help='Directory of simulation data with RFID information')
    parser.add_argument('--real_data_dir', type=str, default='/scratch/tshu2/jyu197/Focus_processed_multi_angle_rfid_real',
                        help='Directory of real-world data with RFID information')
    parser.add_argument('--use_multi_angle', action='store_true', default=True, 
                        help='Use data from all angles (0, 90, 180, 270) instead of just 90 degrees')
    parser.add_argument('--use_noisy', action='store_true', default=True, 
                        help='Use noisy data for training instead of simulated data')
    
    # Dataset selection and splitting strategy
    parser.add_argument('--dataset_type', type=str, choices=['sim', 'real', 'mixed'], default='sim',
                        help='Type of dataset to use: simulated, real-world, or mixed')
    parser.add_argument('--split_strategy', type=str, choices=['default', 'angle-based', 'random-subset'], default='default',
                        help='Strategy for splitting data (default: 70/15/15 random split)')
    parser.add_argument('--train_angles', type=str, nargs='+', default=None,
                        help='Angles to use for training with angle-based splitting (e.g., 0 90)')
    parser.add_argument('--val_angle', type=str, default=None,
                        help='Angle to use for validation with angle-based splitting (e.g., 180)')
    parser.add_argument('--samples_per_class', type=int, default=None,
                        help='Maximum samples per class for angle-based or random-subset splitting')
    parser.add_argument('--real_split_strategy', type=str, choices=['default', 'angle-based', 'random-subset'], default='random-subset',
                        help='Strategy for splitting real data')
    parser.add_argument('--real_val_angle', type=str, default=None,
                        help='Angle to use for real validation data with angle-based splitting')
    parser.add_argument('--real_samples_per_class', type=int, default=37,
                        help='Maximum real samples per class for training')
    
    # Transfer learning parameters
    parser.add_argument('--pretrained_path', type=str, 
                        default="/scratch/tshu2/jyu197/XRF55-repo/hoi_model_checkpoints_loss_coef_0.3_neuron_num_64/best.pth.tar",
                        help='Path to pretrained model for transfer learning')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Use pretrained model and finetune on real/mixed data')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Freeze encoder layers during finetuning (only train the classifier head)')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=60, 
                        help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='Number of workers for data loading')
    
    # Hardware and execution parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./hoi_model_checkpoints_loss_coef_0.3_neuron_num_64', 
                        help='Directory to save checkpoints')
    parser.add_argument('--use_bf16', action='store_true', default=True, 
                        help='Use bfloat16 precision if available')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--hyp_search', action='store_true', default=False, 
                        help='Perform hyperparameter search before training')
    
    args = parser.parse_args()
    
    # Process train_angles to ensure they're in the correct format
    if args.train_angles is not None:
        args.train_angles = [angle for angle in args.train_angles]
    
    return args

def save_plot(train_losses, val_losses, train_action_accs, val_action_accs, train_obj_accs, val_obj_accs, args):
    """Save training and validation metrics plots"""
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot action accuracy
    plt.subplot(2, 2, 2)
    plt.plot(train_action_accs, label='Train Action Accuracy')
    plt.plot(val_action_accs, label='Val Action Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Action Recognition Accuracy')
    
    # Plot object accuracy
    plt.subplot(2, 2, 3)
    plt.plot(train_obj_accs, label='Train Object Accuracy')
    plt.plot(val_obj_accs, label='Val Object Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Object Recognition Accuracy')
    
    # Plot combined accuracy (average of action and object)
    plt.subplot(2, 2, 4)
    train_combined = [(a + o) / 2 for a, o in zip(train_action_accs, train_obj_accs)]
    val_combined = [(a + o) / 2 for a, o in zip(val_action_accs, val_obj_accs)]
    plt.plot(train_combined, label='Train Combined Accuracy')
    plt.plot(val_combined, label='Val Combined Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Combined Accuracy (Action + Object) / 2')
    
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
        best_loss_coef: Best loss coefficient for balancing action and object losses
    """
    # Hyperparameter search space
    learning_rates = [1e-5, 5e-4, 1e-4]
    batch_sizes = [32]
    loss_coefs = [0.3, 0.5, 0.7]  # Object loss coefficients
    
    # Create results directory for the search
    search_dir = os.path.join(args.checkpoint_dir, 'hyperparam_search')
    if not os.path.exists(search_dir):
        os.makedirs(search_dir)
        
    # Initialize best values
    best_val_metric = 0.0  # Will be the average of action and object accuracies
    best_lr = None
    best_batch_size = None
    best_loss_coef = None
    
    # Number of epochs for each configuration
    search_epochs = 3
    
    # Store results for plotting
    results = []
    
    # Determine data type based on arguments
    data_type = "noisy" if args.use_noisy else "sim"
    logger.info(f"Hyperparameter search using {data_type} data")
    
    # Try all combinations
    combos = list(itertools.product(learning_rates, batch_sizes, loss_coefs))
    logger.info(f"Starting hyperparameter search with {len(combos)} combinations")
    logger.info(f"Search space: LR={learning_rates}, BS={batch_sizes}, LOSS_COEF={loss_coefs}")
    
    for i, (lr, bs, loss_coef) in enumerate(combos):
        logger.info(f"Testing combination {i+1}/{len(combos)}: LR={lr}, BS={bs}, LOSS_COEF={loss_coef}")
        
        # Create dataloaders with current batch size
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=bs, 
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=rfid_collate_fn,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=bs,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=rfid_collate_fn,
            pin_memory=True
        )
        
        # Get the number of action classes and object classes from the dataset
        action_num = len(train_dataset.action_types)
        obj_num = 6  # Fixed at 6 for the RFID dataset
        
        # Create model with current configuration
        model = HOINet(
            action_num=action_num,
            obj_num=obj_num,
            dropout_rate=args.dropout_rate,
            num_antennas=args.num_antennas,
            data_format='processed'
        )
        model = model.to(device)
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        
        # Training loop for search epochs
        val_metrics_history = []
        
        for epoch in range(search_epochs):
            # Train one epoch
            model.train()
            epoch_loss = 0
            epoch_samples = 0
            
            with tqdm(total=len(train_dataloader), desc=f"Search [{i+1}/{len(combos)}] Epoch {epoch+1}/{search_epochs}") as pbar:
                for batch_idx, batch in enumerate(train_dataloader):
                    # Move data to device based on data type selected
                    if args.use_noisy:
                        mm_data = [
                            batch['noisy_radar_data'][0].to(device),  # RDspecs
                            batch['noisy_radar_data'][1].to(device),  # AoAspecs
                            batch['noisy_radar_data'][2].to(device)   # specs_mask
                        ]
                        rfid_data = [
                            batch['noisy_rfid_data'][0].to(device),   # obj_loc
                            batch['noisy_rfid_data'][1].to(device)    # obj_mask
                        ]
                    else:
                        mm_data = [
                            batch['radar_data'][0].to(device),  # RDspecs
                            batch['radar_data'][1].to(device),  # AoAspecs
                            batch['radar_data'][2].to(device)   # specs_mask
                        ]
                        rfid_data = [
                            batch['rfid_data'][0].to(device),   # obj_loc
                            batch['rfid_data'][1].to(device)    # obj_mask
                        ]
                    
                    action_labels = batch['action_labels'].to(device)
                    obj_labels = batch['obj_labels'].to(device)
                    
                    # Forward pass with mixed precision if available
                    if args.use_bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                            action_output, obj_output = model(mm_data, rfid_data)
                            batch_loss = loss_fn(action_output, action_labels, obj_output, obj_labels, loss_coef)
                    else:
                        action_output, obj_output = model(mm_data, rfid_data)
                        batch_loss = loss_fn(action_output, action_labels, obj_output, obj_labels, loss_coef)
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    epoch_loss += batch_loss.item() * action_labels.size(0)
                    epoch_samples += action_labels.size(0)
                    
                    pbar.update(1)
                    pbar.set_postfix({'loss': batch_loss.item()})
            
            # Evaluate on validation set
            val_loss, val_metrics = evaluate(model, val_dataloader, device, args, loss_coef)
            val_action_acc = val_metrics['Action Accuracy']
            val_obj_acc = val_metrics['Object Accuracy']
            val_avg_acc = (val_action_acc + val_obj_acc) / 2  # Combined metric
            
            val_metrics_history.append(val_avg_acc)
            
            logger.info(f"Search [{i+1}/{len(combos)}] Epoch {epoch+1}/{search_epochs} - "
                       f"Train Loss: {epoch_loss/epoch_samples:.4f}, "
                       f"Val Action Acc: {val_action_acc:.4f}, "
                       f"Val Object Acc: {val_obj_acc:.4f}, "
                       f"Val Avg Acc: {val_avg_acc:.4f}")
        
        # Calculate final validation metric for this configuration (average of last 2 epochs if available)
        if len(val_metrics_history) > 1:
            final_val_metric = sum(val_metrics_history[-2:]) / 2
        else:
            final_val_metric = val_metrics_history[-1]
            
        # Store result
        results.append({
            'lr': lr,
            'batch_size': bs,
            'loss_coef': loss_coef,
            'val_metric': final_val_metric
        })
        
        # Update best configuration if needed
        if final_val_metric > best_val_metric:
            best_val_metric = final_val_metric
            best_lr = lr
            best_batch_size = bs
            best_loss_coef = loss_coef
            
        # Clear memory
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Print and save results
    logger.info("\nHyperparameter Search Results:")
    results_file = os.path.join(search_dir, 'search_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Dataset type: processed with RFID, Data type: {data_type}\n\n")
        for result in sorted(results, key=lambda x: x['val_metric'], reverse=True):
            result_str = f"LR: {result['lr']}, BS: {result['batch_size']}, LOSS_COEF: {result['loss_coef']}, Val Metric: {result['val_metric']:.4f}"
            logger.info(result_str)
            f.write(result_str + '\n')
    
    logger.info(f"\nBest Configuration: LR={best_lr}, BS={best_batch_size}, LOSS_COEF={best_loss_coef}, Val Metric={best_val_metric:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot for different batch sizes
    plt.subplot(1, 2, 1)
    for bs in batch_sizes:
        for lc in loss_coefs:
            bs_lc_results = [r for r in results if r['batch_size'] == bs and r['loss_coef'] == lc]
            bs_lc_results = sorted(bs_lc_results, key=lambda x: x['lr'])
            if bs_lc_results:  # Only plot if there are results for this combination
                plt.plot([r['lr'] for r in bs_lc_results], 
                         [r['val_metric'] for r in bs_lc_results], 
                         marker='o', 
                         label=f'BS={bs}, LC={lc}')
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Metric (Avg Accuracy)')
    plt.title('Search Results by Learning Rate')
    plt.legend()
    plt.grid(True)
    
    # Plot for different loss coefficients
    plt.subplot(1, 2, 2)
    for bs in batch_sizes:
        for lr in learning_rates:
            bs_lr_results = [r for r in results if r['batch_size'] == bs and r['lr'] == lr]
            bs_lr_results = sorted(bs_lr_results, key=lambda x: x['loss_coef'])
            if bs_lr_results:  # Only plot if there are results
                plt.plot([r['loss_coef'] for r in bs_lr_results], 
                         [r['val_metric'] for r in bs_lr_results], 
                         marker='o', 
                         label=f'BS={bs}, LR={lr}')
    
    plt.xlabel('Loss Coefficient')
    plt.ylabel('Validation Metric (Avg Accuracy)')
    plt.title('Search Results by Loss Coefficient')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(search_dir, 'search_results.png'))
    
    return best_lr, best_batch_size, best_loss_coef
    
def train_and_evaluate(model, train_dataset, val_dataset, optimizer, scheduler, args, loss_coef):
    """Train the model and evaluate on validation set"""
    
    global BEST_ACTION_ACC
    global BEST_OBJ_ACC
    
    # Create dataloaders based on dataset type
    if isinstance(train_dataset, tuple):
        # Mixed dataset (sim + real) approach
        sim_dataset, real_dataset = train_dataset
        logger.info(f"Using mixed datasets: {len(sim_dataset)} sim + {len(real_dataset)} real samples")
        train_dataloader = MixedDataLoader(
            sim_dataset, 
            real_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=rfid_collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=rfid_collate_fn,
            pin_memory=True
        )
        
        # Track separate losses for sim and real data
        sim_losses = []
        real_losses = []
    else:
        # Single dataset approach
        data_type = args.dataset_type
        logger.info(f"Training using {data_type} dataset")
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=rfid_collate_fn,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=rfid_collate_fn,
            pin_memory=True
        )
    
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
    best_val_metric = 0.0  # Will be the average of action and object accuracies
    train_losses = []
    val_losses = []
    train_action_accs = []
    val_action_accs = []
    train_obj_accs = []
    val_obj_accs = []
    
    # For mixed datasets
    epoch_sim_loss = 0
    epoch_real_loss = 0
    epoch_sim_samples = 0
    epoch_real_samples = 0
    
    # Starting epoch
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best.pth.tar')
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = model_utils.load_checkpoint(checkpoint_path, model, optimizer)
            start_epoch = checkpoint['epoch']
            best_val_metric = checkpoint.get('val_metric', 0.0)
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            train_action_accs = checkpoint.get('train_action_accs', [])
            val_action_accs = checkpoint.get('val_action_accs', [])
            train_obj_accs = checkpoint.get('train_obj_accs', [])
            val_obj_accs = checkpoint.get('val_obj_accs', [])
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
        epoch_sim_loss = 0
        epoch_real_loss = 0
        epoch_sim_samples = 0
        epoch_real_samples = 0
        all_action_preds = []
        all_action_labels = []
        all_obj_preds = []
        all_obj_labels = []
        
        # Use tqdm for progress bar
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]") as pbar:
            for batch_idx, batch_data in enumerate(train_dataloader):
                # Handle mixed dataset with separate batch structure
                if isinstance(train_dataset, tuple):
                    batch, batch_type = batch_data
                    is_sim_batch = (batch_type == 'sim')
                else:
                    batch = batch_data
                    is_sim_batch = False
                
                # Process data based on batch type (noisy or clean)
                if args.use_noisy:
                    mm_data = [
                        batch['noisy_radar_data'][0].to(device),  # RDspecs
                        batch['noisy_radar_data'][1].to(device),  # AoAspecs
                        batch['noisy_radar_data'][2].to(device)   # specs_mask
                    ]
                    rfid_data = [
                        batch['noisy_rfid_data'][0].to(device),   # obj_loc
                        batch['noisy_rfid_data'][1].to(device)    # obj_mask
                    ]
                else:
                    mm_data = [
                        batch['radar_data'][0].to(device),  # RDspecs
                        batch['radar_data'][1].to(device),  # AoAspecs
                        batch['radar_data'][2].to(device)   # specs_mask
                    ]
                    rfid_data = [
                        batch['rfid_data'][0].to(device),   # obj_loc
                        batch['rfid_data'][1].to(device)    # obj_mask
                    ]
                
                action_labels = batch['action_labels'].to(device)
                obj_labels = batch['obj_labels'].to(device)
                
                # Convert to bfloat16 if using mixed precision
                if args.use_bf16 and torch.cuda.get_device_capability()[0] >= 8:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # Forward pass
                        action_output, obj_output = model(mm_data, rfid_data)
                        batch_loss = loss_fn(action_output, action_labels, obj_output, obj_labels, loss_coef)
                else:
                    # Forward pass (standard precision)
                    action_output, obj_output = model(mm_data, rfid_data)
                    batch_loss = loss_fn(action_output, action_labels, obj_output, obj_labels, loss_coef)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                # Update metrics
                batch_size = action_labels.size(0)
                epoch_loss += batch_loss.item() * batch_size
                epoch_samples += batch_size
                
                # Track separate losses for sim and real data in mixed mode
                if isinstance(train_dataset, tuple):
                    if is_sim_batch:
                        epoch_sim_loss += batch_loss.item() * batch_size
                        epoch_sim_samples += batch_size
                    else:
                        epoch_real_loss += batch_loss.item() * batch_size
                        epoch_real_samples += batch_size
                
                # Track predictions
                _, action_preds = torch.max(action_output, 1)
                all_action_preds.extend(action_preds.cpu().numpy())
                all_action_labels.extend(action_labels.cpu().numpy())
                
                _, obj_preds = torch.max(obj_output, 1)
                all_obj_preds.extend(obj_preds.cpu().numpy())
                all_obj_labels.extend(obj_labels.cpu().numpy())
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': batch_loss.item()})
        
        # Calculate epoch metrics
        train_loss = epoch_loss / epoch_samples
        train_action_acc = np.mean(np.array(all_action_preds) == np.array(all_action_labels))
        train_obj_acc = np.mean(np.array(all_obj_preds) == np.array(all_obj_labels))
        
        # Track separate metrics for mixed dataset
        if isinstance(train_dataset, tuple) and epoch_sim_samples > 0 and epoch_real_samples > 0:
            sim_loss = epoch_sim_loss / epoch_sim_samples
            real_loss = epoch_real_loss / epoch_real_samples
            sim_losses.append(sim_loss)
            real_losses.append(real_loss)
            
            logger.info(f"Sim Loss: {sim_loss:.4f}, Real Loss: {real_loss:.4f}")
            
            # Update mixing weights for next epoch
            train_dataloader.update_weights(sim_loss, real_loss)
        
        # Evaluate on validation set
        val_loss, val_metrics = evaluate(model, val_dataloader, device, args, loss_coef)
        val_action_acc = val_metrics['Action Accuracy']
        val_obj_acc = val_metrics['Object Accuracy']
        
        # Calculate combined metric (average of action and object accuracies)
        val_metric = (val_action_acc + val_obj_acc) / 2
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Train Action Acc: {train_action_acc:.4f}, "
                  f"Train Obj Acc: {train_obj_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Action Acc: {val_action_acc:.4f}, "
                  f"Val Obj Acc: {val_obj_acc:.4f}")
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_action_accs.append(train_action_acc)
        val_action_accs.append(val_action_acc)
        train_obj_accs.append(train_obj_acc)
        val_obj_accs.append(val_obj_acc)
        
        # Save checkpoint
        is_best = val_metric > best_val_metric
        if is_best:
            best_val_metric = val_metric
            
        # Checkpoint state
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_action_accs': train_action_accs,
            'val_action_accs': val_action_accs,
            'train_obj_accs': train_obj_accs,
            'val_obj_accs': val_obj_accs,
            'val_metric': val_metric,
            'encoder_dim': args.encoder_dim,
            'fusion_dim': args.fusion_dim,
            'neuron_num': args.neuron_num,
            'loss_coef': loss_coef,
            'use_noisy': args.use_noisy,
            'use_multi_angle': args.use_multi_angle,
            'dropout_rate': args.dropout_rate,
            'num_antennas': args.num_antennas,
            'dataset_type': args.dataset_type
        }
        
        model_utils.save_checkpoint(state, is_best, args.checkpoint_dir)
        
        # Save training plots
        save_plot(train_losses, val_losses, train_action_accs, val_action_accs, train_obj_accs, val_obj_accs, args)
    
    logger.info(f"Training completed. Best validation metric: {best_val_metric:.4f}")
    BEST_ACTION_ACC = np.max(train_action_accs)
    BEST_OBJ_ACC = np.max(train_obj_accs)
    
def evaluate(model, dataloader, device, args, loss_coef):
    """Evaluate the model on the given dataloader"""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    all_action_outputs = []
    all_action_labels = []
    all_obj_outputs = []
    all_obj_labels = []
    
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluation") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                # Move data to device based on data type selected
                if args.use_noisy:
                    mm_data = [
                        batch['noisy_radar_data'][0].to(device),  # RDspecs
                        batch['noisy_radar_data'][1].to(device),  # AoAspecs
                        batch['noisy_radar_data'][2].to(device)   # specs_mask
                    ]
                    rfid_data = [
                        batch['noisy_rfid_data'][0].to(device),   # obj_loc
                        batch['noisy_rfid_data'][1].to(device)    # obj_mask
                    ]
                else:
                    mm_data = [
                        batch['radar_data'][0].to(device),  # RDspecs
                        batch['radar_data'][1].to(device),  # AoAspecs
                        batch['radar_data'][2].to(device)   # specs_mask
                    ]
                    rfid_data = [
                        batch['rfid_data'][0].to(device),   # obj_loc
                        batch['rfid_data'][1].to(device)    # obj_mask
                    ]
                
                action_labels = batch['action_labels'].to(device)
                obj_labels = batch['obj_labels'].to(device)
                
                # Forward pass with bfloat16 if specified
                if args.use_bf16 and torch.cuda.get_device_capability()[0] >= 8:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        action_output, obj_output = model(mm_data, rfid_data)
                        batch_loss = loss_fn(action_output, action_labels, obj_output, obj_labels, loss_coef)
                else:
                    action_output, obj_output = model(mm_data, rfid_data)
                    batch_loss = loss_fn(action_output, action_labels, obj_output, obj_labels, loss_coef)
                
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

def get_datasets(args):
    """
    Create datasets based on the specified type and splitting strategy
    
    Args:
        args: Command line arguments containing dataset configuration
        
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    if args.dataset_type == 'sim':
        # Simulated data only
        logger.info(f"Using simulated data with {args.split_strategy} splitting strategy")
        
        train_dataset = FocusDatasetwithRFID(
            base_dir=args.data_dir,
            split='train',
            use_multi_angle=args.use_multi_angle,
            use_noisy=args.use_noisy,
            split_strategy=args.split_strategy,
            train_angles=args.train_angles,
            val_angle=args.val_angle,
            samples_per_class=args.samples_per_class
        )
        
        val_dataset = FocusDatasetwithRFID(
            base_dir=args.data_dir,
            split='val',
            use_multi_angle=args.use_multi_angle,
            use_noisy=args.use_noisy,
            split_strategy=args.split_strategy,
            train_angles=args.train_angles,
            val_angle=args.val_angle,
            samples_per_class=args.samples_per_class
        )
        
    elif args.dataset_type == 'real':
        # Real-world data only
        logger.info(f"Using real-world data with {args.real_split_strategy} splitting strategy")
        
        train_dataset = FocusRealDatasetwithRFID(
            base_dir=args.real_data_dir,
            split='train',
            use_multi_angle=args.use_multi_angle,
            random_subset_n=args.real_samples_per_class,
            val_angle=args.real_val_angle
        )
        
        val_dataset = FocusRealDatasetwithRFID(
            base_dir=args.real_data_dir,
            split='val',
            use_multi_angle=args.use_multi_angle,
            random_subset_n=args.real_samples_per_class,
            val_angle=args.real_val_angle
        )
        
    elif args.dataset_type == 'mixed':
        # Combining real and simulated data for fine-tuning
        logger.info("Using mixed dataset (real + simulated)")
        
        # Get simulated dataset
        sim_train_dataset = FocusDatasetwithRFID(
            base_dir=args.data_dir,
            split='train',
            use_multi_angle=args.use_multi_angle,
            use_noisy=args.use_noisy,
            split_strategy=args.split_strategy,
            train_angles=args.train_angles,
            val_angle=args.val_angle,
            samples_per_class=args.samples_per_class
        )
        
        # Get real dataset
        real_train_dataset = FocusRealDatasetwithRFID(
            base_dir=args.real_data_dir,
            split='train',
            use_multi_angle=args.use_multi_angle,
            random_subset_n=args.real_samples_per_class,
            val_angle=args.real_val_angle
        )
        
        # For validation, we use real data to match the deployment scenario
        val_dataset = FocusRealDatasetwithRFID(
            base_dir=args.real_data_dir,
            split='val',
            use_multi_angle=args.use_multi_angle,
            random_subset_n=args.real_samples_per_class,
            val_angle=args.real_val_angle
        )
        
        # Combine datasets
        train_dataset = (sim_train_dataset, real_train_dataset)
    
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    logger.info(f"Train dataset size: {len(train_dataset) if not isinstance(train_dataset, tuple) else f'{len(train_dataset[0])} sim + {len(train_dataset[1])} real'}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def load_pretrained_model(model, pretrained_path, freeze_encoder=False):
    """
    Load a pretrained model for fine-tuning.
    
    Args:
        model: Model to load pretrained weights into
        pretrained_path: Path to the pretrained model file
        freeze_encoder: Whether to freeze encoder layers
        
    Returns:
        model: Model with pretrained weights
    """
    logger.info(f"Loading pretrained model from {pretrained_path}")
    
    if not os.path.exists(pretrained_path):
        logger.warning(f"Pretrained model not found at {pretrained_path}")
        return model
    
    # Load checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    # Handle checkpoint format variations
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict into model
    model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained model loaded successfully")
    
    # Freeze encoder layers if specified
    if freeze_encoder:
        logger.info("Freezing encoder layers")
        for name, param in model.named_parameters():
            if 'encoder' in name or 'feature_extractor' in name:
                param.requires_grad = False
                logger.debug(f"Froze parameter: {name}")
    
    return model

class MixedDataLoader:
    """Custom dataloader to interleave batches from simulated and real datasets"""
    def __init__(self, sim_dataset, real_dataset, batch_size, shuffle=True, num_workers=1, collate_fn=None, drop_last=False):
        self.sim_loader = DataLoader(
            sim_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=drop_last
        )
        
        self.real_loader = DataLoader(
            real_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=drop_last
        )
        
        # Calculate length - use the longer dataloader
        self.length = max(len(self.sim_loader), len(self.real_loader))
        
        # Create iterators
        self.sim_iter = iter(self.sim_loader)
        self.real_iter = iter(self.real_loader)
        
        # Adaptive mixing weights
        self.sim_weight = 0.5  # Start with equal weighting
        
    def __iter__(self):
        # Reset iterators at the start of each epoch
        self.sim_iter = iter(self.sim_loader)
        self.real_iter = iter(self.real_loader)
        return self
    
    def __next__(self):
        # Weighted random selection of dataset
        if np.random.random() < self.sim_weight:
            try:
                return next(self.sim_iter), 'sim'
            except StopIteration:
                # If sim is exhausted, reset it and return real
                self.sim_iter = iter(self.sim_loader)
                return next(self.real_iter), 'real'
        else:
            try:
                return next(self.real_iter), 'real'
            except StopIteration:
                # If real is exhausted, reset it and return sim
                self.real_iter = iter(self.real_loader)
                return next(self.sim_iter), 'sim'
    
    def __len__(self):
        return self.length
    
    def update_weights(self, sim_loss, real_loss):
        """Adaptively update mixing weights based on relative loss magnitudes"""
        total = sim_loss + real_loss
        if total > 0:
            self.sim_weight = real_loss / total  # More weight to dataset with higher loss

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    model_utils.setup_seed(args.seed)
    
    # Update model constants
    # Note: This requires importing and modifying the transformer_model_with_rfid module
    from model import transformer_model_with_rfid
    transformer_model_with_rfid.ENCODER_DIM = args.encoder_dim
    transformer_model_with_rfid.FUSION_DIM = args.fusion_dim
    transformer_model_with_rfid.NEURON_NUM = args.neuron_num
    
    # Log model settings
    logger.info(f"Model settings: ENCODER_DIM={args.encoder_dim}, FUSION_DIM={args.fusion_dim}, NEURON_NUM={args.neuron_num}")
    logger.info(f"Model settings: dropout={args.dropout_rate}, antennas={args.num_antennas}, loss_coef={args.loss_coef}")
    
    # Log data choices
    logger.info(f"Dataset type: {args.dataset_type}")
    
    if args.dataset_type == 'sim':
        logger.info(f"Split strategy: {args.split_strategy}")
        if args.split_strategy == 'angle-based':
            logger.info(f"Train angles: {args.train_angles}, Val angle: {args.val_angle}")
        if args.samples_per_class:
            logger.info(f"Samples per class: {args.samples_per_class}")
    elif args.dataset_type == 'real' or args.dataset_type == 'mixed':
        logger.info(f"Real data split strategy: {args.real_split_strategy}")
        if args.real_val_angle:
            logger.info(f"Real val angle: {args.real_val_angle}")
        logger.info(f"Real samples per class: {args.real_samples_per_class}")
    
    # Log finetuning settings if applicable
    if args.finetune:
        logger.info(f"Finetuning from pretrained model: {args.pretrained_path}")
        logger.info(f"Freezing encoder: {args.freeze_encoder}")
    
    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets based on the specified type
    train_dataset, val_dataset = get_datasets(args)
    
    # Perform hyperparameter search if specified
    loss_coef = args.loss_coef  # Default value
    if args.hyp_search:
        logger.info("Starting hyperparameter search")
        # Handle mixed dataset case for hyperparameter search
        if isinstance(train_dataset, tuple):
            # Just use sim dataset for hyperparameter search
            search_train_dataset = train_dataset[0]
        else:
            search_train_dataset = train_dataset
        
        best_lr, best_batch_size, best_loss_coef = hyperparameter_search(search_train_dataset, val_dataset, device, args)
        
        # Update parameters with best values
        args.learning_rate = best_lr
        args.batch_size = best_batch_size
        loss_coef = best_loss_coef
        logger.info(f"Using best hyperparameters: LR={best_lr}, BS={best_batch_size}, LOSS_COEF={best_loss_coef}")
    else:
        logger.info(f"Using default hyperparameters: LR={args.learning_rate}, BS={args.batch_size}, LOSS_COEF={loss_coef}")
    
    # Get the number of action classes from the dataset
    if isinstance(train_dataset, tuple):
        # Mixed dataset - either should have the same number of classes
        action_num = len(train_dataset[0].action_types)
    else:
        action_num = len(train_dataset.action_types)
    
    obj_num = 6  # Fixed at 6 for the RFID dataset
    logger.info(f"Number of action classes: {action_num}")
    logger.info(f"Number of object classes: {obj_num}")
    
    # Create model
    model = HOINet(
        action_num=action_num,
        obj_num=obj_num,
        dropout_rate=args.dropout_rate,
        num_antennas=args.num_antennas,
        data_format='processed'
    )
    model = model.to(device)
    
    # Load pretrained model for fine-tuning if specified
    if args.finetune:
        model = load_pretrained_model(model, args.pretrained_path, args.freeze_encoder)
    
    # Print model summary
    model_size = getModelSize(model)
    logger.info(f"Model parameters: {model_size[1]:,}")
    logger.info(f"Model size: {model_size[-1]:.2f} MB")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 45], gamma=0.1)
    
    # Record training configuration
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    with open(os.path.join(args.checkpoint_dir, 'training_config.txt'), 'w') as f:
        f.write(f"Dataset type: {args.dataset_type}\n")
        f.write(f"Data type: {args.use_noisy and 'noisy' or 'clean'}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Loss coefficient: {loss_coef}\n")
        f.write(f"BFloat16: {args.use_bf16}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of epochs: {args.num_epochs}\n")
        f.write(f"Using multi-angle data: {args.use_multi_angle}\n")
        f.write(f"Encoder dimension: {args.encoder_dim}\n")
        f.write(f"Fusion dimension: {args.fusion_dim}\n")
        f.write(f"Neuron number (RFID): {args.neuron_num}\n")
        f.write(f"Dropout rate: {args.dropout_rate}\n")
        f.write(f"Number of antennas: {args.num_antennas}\n")
        f.write(f"Model size: {model_size[-1]:.2f} MB\n")
        f.write(f"Model parameters: {model_size[1]:,}\n")
        f.write(f"Action classes: {action_num}\n")
        f.write(f"Object classes: {obj_num}\n")
        
        if args.dataset_type == 'sim':
            f.write(f"Split strategy: {args.split_strategy}\n")
            if args.split_strategy == 'angle-based':
                f.write(f"Train angles: {args.train_angles}\n")
                f.write(f"Val angle: {args.val_angle}\n")
            if args.samples_per_class:
                f.write(f"Samples per class: {args.samples_per_class}\n")
        elif args.dataset_type in ['real', 'mixed']:
            f.write(f"Real data split strategy: {args.real_split_strategy}\n")
            if args.real_val_angle:
                f.write(f"Real val angle: {args.real_val_angle}\n")
            f.write(f"Real samples per class: {args.real_samples_per_class}\n")
            
        if args.finetune:
            f.write(f"Finetuning from: {args.pretrained_path}\n")
            f.write(f"Froze encoder: {args.freeze_encoder}\n")
    
    # Train and evaluate
    train_and_evaluate(model, train_dataset, val_dataset, optimizer, scheduler, args, loss_coef)

    with open(os.path.join(args.checkpoint_dir, 'training_config.txt'), 'a') as f:
        f.write(f"Best action accuracy: {BEST_ACTION_ACC:.4f}\n")
        f.write(f"Best object accuracy: {BEST_OBJ_ACC:.4f}\n")

if __name__ == "__main__":
    main() 