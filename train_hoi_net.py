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
import random

# Import custom modules
from model.transformer_model_with_rfid_no_focus import HOINet as HOINet_NoFocus
from model.transformer_model_with_rfid import HOINet as HOINet_Focus
from model.transformer_model_with_rfid_no_focus import loss_fn, classifer_metrics, getModelSize
import model_utils
from FocusDataset import DatasetwithRFID, new_collate_fn, RealDatasetwithRFID, FocusDatasetwithRFID, rfid_collate_fn, FocusRealDatasetwithRFID

BEST_ACTION_ACC = 0.0
BEST_OBJ_ACC = 0.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('HOINet_Training')

def parse_args():
    parser = argparse.ArgumentParser(description='HOI Recognition Training with Radar and RFID Data')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, choices=['focus', 'no_focus'], default='no_focus',
                        help='Which model architecture to use (focus or no_focus)')
    parser.add_argument('--encoder_dim', type=int, default=256, 
                        help='Dimension of encoder models')
    parser.add_argument('--fusion_dim', type=int, default=512, 
                        help='Dimension for fusion transformer')
    parser.add_argument('--neuron_num', type=int, default=64, 
                        help='Dimension for object branch')
    parser.add_argument('--num_antennas', type=int, default=12, 
                        help='Number of antennas in radar data (only used for focus model)')
    parser.add_argument('--dropout_rate', type=float, default=0.2, 
                        help='Dropout rate for all models')
    parser.add_argument('--loss_coef', type=float, default=0.5,
                        help='Weight of the object classification loss')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='/weka/scratch/rzhao36/lwang/datasets/HOI/datasets/classic', # '/scratch/tshu2/jyu197/Focus_processed_multi_angle_rfid'
                        help='Directory of simulation data with RFID information')
    parser.add_argument('--real_data_dir', type=str, default='/weka/scratch/rzhao36/lwang/datasets/HOI/RealAction/datasets/classic', # '/scratch/tshu2/jyu197/Focus_processed_multi_angle_rfid_real'
                        help='Directory of real-world data with RFID information')
    parser.add_argument('--use_multi_angle', action='store_true', default=True, 
                        help='Use data from all angles (0, 90, 180, 270) instead of just 90 degrees')
    parser.add_argument('--use_noisy', action='store_true', default=True, 
                        help='Use noisy data for training instead of simulated data')
    
    # Dataset selection and splitting strategy
    parser.add_argument('--dataset_type', type=str, choices=['sim', 'real', 'mixed'], default='real',
                        help='Type of dataset to use: simulated, real-world, or mixed')
    parser.add_argument('--split_strategy', type=str, choices=['default', 'angle-based', 'random-subset'], default='random-subset',
                        help='Strategy for splitting data (default: 70/15/15 random split)')
    parser.add_argument('--train_angles', type=str, nargs='+', default=None,
                        help='Angles to use for training with angle-based splitting (e.g., 0 90)')
    parser.add_argument('--val_angle', type=str, default=None,
                        help='Angle to use for validation with angle-based splitting (e.g., 180)')
    parser.add_argument('--samples_per_class', type=int, default=27,
                        help='Maximum samples per class for angle-based or random-subset splitting')
    parser.add_argument('--real_split_strategy', type=str, choices=['default', 'angle-based', 'random-subset'], default='random-subset',
                        help='Strategy for splitting real data')
    parser.add_argument('--real_val_angle', type=str, default=None,
                        help='Angle to use for real validation data with angle-based splitting')
    parser.add_argument('--real_samples_per_class', type=int, default=27,
                        help='Maximum real samples per class for training')
    
    # Transfer learning parameters
    parser.add_argument('--pretrained_path', type=str, 
                        default="/scratch/tshu2/jyu197/XRF55-repo/hoi_model_pretrained_classic_data/best.pth.tar",
                        help='Path to pretrained model for transfer learning')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Use pretrained model and finetune on real/mixed data')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Freeze encoder layers during finetuning (only train the classifier head)')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='Number of workers for data loading')
    
    # Hardware and execution parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./hoi_model_from_scratch_lw_less_real', 
                        help='Directory to save checkpoints')
    parser.add_argument('--use_bf16', action='store_true', default=True, 
                        help='Use bfloat16 precision if available')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=39, 
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
        
        if args.model_type == 'focus':
            # Use Focus datasets for Focus model
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
        else:
            # Use no-focus datasets for no-focus model
            train_dataset = DatasetwithRFID(
                base_dir=args.data_dir,
                split='train',
                use_multi_angle=args.use_multi_angle,
                use_noisy=args.use_noisy,
                split_strategy=args.split_strategy,
                train_angles=args.train_angles,
                val_angle=args.val_angle,
                samples_per_class=args.samples_per_class
            )
            
            val_dataset = DatasetwithRFID(
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
        
        if args.model_type == 'focus':
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
        else:
            train_dataset = RealDatasetwithRFID(
                base_dir=args.real_data_dir,
                split='train',
                use_multi_angle=args.use_multi_angle,
                random_subset_n=args.real_samples_per_class,
                val_angle=args.real_val_angle
            )
            
            val_dataset = RealDatasetwithRFID(
                base_dir=args.real_data_dir,
                split='val',
                use_multi_angle=args.use_multi_angle,
                random_subset_n=args.real_samples_per_class,
                val_angle=args.real_val_angle
            )
        
    elif args.dataset_type == 'mixed':
        # Combining real and simulated data for fine-tuning
        logger.info("Using mixed dataset (real + simulated)")
        
        if args.model_type == 'focus':
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
        else:
            # Get simulated dataset
            sim_train_dataset = DatasetwithRFID(
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
            real_train_dataset = RealDatasetwithRFID(
                base_dir=args.real_data_dir,
                split='train',
                use_multi_angle=args.use_multi_angle,
                random_subset_n=args.real_samples_per_class,
                val_angle=args.real_val_angle
            )
            
            # For validation, we use real data to match the deployment scenario
            val_dataset = RealDatasetwithRFID(
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
    Load a pretrained model for transfer learning
    
    Args:
        model: Model to load weights into
        pretrained_path: Path to pretrained model checkpoint
        freeze_encoder: Whether to freeze encoder layers
        
    Returns:
        model: Model with loaded weights
    """
    logger.info(f"Loading pretrained model from {pretrained_path}")
    if not os.path.exists(pretrained_path):
        logger.warning(f"Pretrained model not found at {pretrained_path}, skipping loading")
        return model
    
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)  # Handle different checkpoint formats
        
        # Filter out classifier if necessary
        model_dict = model.state_dict()
        
        # Handle keys that might have 'module.' prefix due to DataParallel
        fixed_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                fixed_state_dict[k[7:]] = v
            else:
                fixed_state_dict[k] = v
        
        # Update state_dict to only include keys in model_dict
        fixed_state_dict = {k: v for k, v in fixed_state_dict.items() if k in model_dict}
        
        # Check missing keys
        missing_keys = set(model_dict.keys()) - set(fixed_state_dict.keys())
        if missing_keys:
            logger.warning(f"Missing keys in pretrained model: {missing_keys}")
        
        # Load weights
        model.load_state_dict(fixed_state_dict, strict=False)
        logger.info(f"Successfully loaded pretrained model")
        
        # Freeze specific encoder components if requested
        if freeze_encoder:
            logger.info("Freezing specific encoder layers")
            
            # Define specific encoder components to freeze
            components_to_freeze = [
                'mmWaveBranch.range_encoder',
                'mmWaveBranch.music_encoder',
                'mmWaveBranch.doppler_encoder',
                'OBJBranch.encoder'
            ]
            
            # Freeze parameters in the specified components
            for name, param in model.named_parameters():
                for component in components_to_freeze:
                    if component in name:
                        param.requires_grad = False
                        logger.debug(f"Froze parameter: {name}")
                        break
            
            # Count and log frozen parameters
            frozen_count = sum(1 for param in model.parameters() if not param.requires_grad)
            total_count = sum(1 for _ in model.parameters())
            logger.info(f"Froze {frozen_count}/{total_count} parameters")
        
        return model
    except Exception as e:
        logger.error(f"Error loading pretrained model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return model

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

    def update_weights(self, *_, **__):
        """Kept for backwards compatibility; no adaptive weights here."""
        pass

def train_and_evaluate(model, train_dataset, val_dataset, optimizer, scheduler, args, loss_coef):
    """Train the model and evaluate on validation set"""
    
    global BEST_ACTION_ACC
    global BEST_OBJ_ACC
    
    # Determine which collate function to use based on model type
    collate_fn = rfid_collate_fn if args.model_type == 'focus' else new_collate_fn
    
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
            collate_fn=collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    else:
        # Single dataset approach
        data_type = args.dataset_type
        logger.info(f"Training using {data_type} dataset")
        
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
    
    # Check for checkpoints directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    # Setup for mixed precision training with bfloat16
    scaler = None
    if args.use_bf16 and torch.cuda.get_device_capability()[0] >= 8:
        logger.info("Using bfloat16 mixed precision training")
        # Note: Scaler not used with bfloat16, just using autocast directly
    
    # Training loop
    device = next(model.parameters()).device
    train_losses = []
    val_losses = []
    train_action_accs = []
    val_action_accs = []
    train_obj_accs = []
    val_obj_accs = []
    best_val_metric = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        model.train()
        
        # Metrics for this epoch
        epoch_loss = 0
        epoch_samples = 0
        epoch_sim_loss = 0
        epoch_sim_samples = 0
        epoch_real_loss = 0
        epoch_real_samples = 0
        
        # For metrics calculation - use rolling accumulation to avoid storing all outputs
        all_action_outputs_list = []
        all_action_labels_list = []
        all_obj_outputs_list = []
        all_obj_labels_list = []
        
        # Use fixed-size tensors for accumulation
        max_batches_to_accumulate = min(50, len(train_dataloader))
        action_output_accum = None
        action_label_accum = None
        obj_output_accum = None
        obj_label_accum = None
        current_batch_count = 0
        
        # Progress bar
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]") as pbar:
            for batch_idx, batch_data in enumerate(train_dataloader):
                # Unpack batch data based on dataset type
                if isinstance(train_dataset, tuple):
                    batch, batch_type = batch_data
                    is_sim_batch = (batch_type == 'sim')
                else:
                    batch = batch_data
                    is_sim_batch = False
                
                # Process data based on batch type (noisy or clean)
                if args.use_noisy:
                    mm_data = batch['noisy_radar_data']  # This is now [padded_radar_data, radar_padding_masks]
                    rfid_data = batch['noisy_rfid_data']  # This is now [padded_rfid_data, rfid_padding_masks]
                else:
                    mm_data = batch['radar_data']  # This is now [padded_radar_data, radar_padding_masks]
                    rfid_data = batch['rfid_data']  # This is now [padded_rfid_data, rfid_padding_masks]
                
                # Move all tensors in mm_data and rfid_data to the selected device
                for i in range(len(mm_data)):
                    mm_data[i] = mm_data[i].to(device)
                for i in range(len(rfid_data)):
                    rfid_data[i] = rfid_data[i].to(device)
                
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
                
                # Accumulate predictions for metrics in a memory-efficient way
                if current_batch_count < max_batches_to_accumulate:
                    # Detach tensors to free computation graph
                    action_output_cpu = action_output.detach().cpu()
                    action_labels_cpu = action_labels.cpu()
                    obj_output_cpu = obj_output.detach().cpu()
                    obj_labels_cpu = obj_labels.cpu()
                    
                    if action_output_accum is None:
                        action_output_accum = action_output_cpu
                        action_label_accum = action_labels_cpu
                        obj_output_accum = obj_output_cpu
                        obj_label_accum = obj_labels_cpu
                    else:
                        action_output_accum = torch.cat([action_output_accum, action_output_cpu], dim=0)
                        action_label_accum = torch.cat([action_label_accum, action_labels_cpu], dim=0)
                        obj_output_accum = torch.cat([obj_output_accum, obj_output_cpu], dim=0)
                        obj_label_accum = torch.cat([obj_label_accum, obj_labels_cpu], dim=0)
                    
                    current_batch_count += 1
                    
                    # If we've reached the limit, compute metrics and reset
                    if current_batch_count == max_batches_to_accumulate or batch_idx == len(train_dataloader) - 1:
                        metrics = classifer_metrics(action_output_accum, action_label_accum, obj_output_accum, obj_label_accum)
                        all_action_outputs_list.append(metrics['Action Accuracy'])
                        all_obj_outputs_list.append(metrics['Object Accuracy'])
                        
                        # Clear accumulation tensors to free memory
                        action_output_accum = None
                        action_label_accum = None
                        obj_output_accum = None
                        obj_label_accum = None
                        current_batch_count = 0
                
                # Clean up to free memory
                del mm_data, rfid_data, action_labels, obj_labels, action_output, obj_output, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Update progress bar with metrics
                avg_loss = epoch_loss / epoch_samples
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
                pbar.update(1)
                
                # Update mixed dataloader weights if applicable
                if isinstance(train_dataloader, MixedDataLoader) and epoch_sim_samples > 0 and epoch_real_samples > 0:
                    sim_avg_loss = epoch_sim_loss / epoch_sim_samples if epoch_sim_samples > 0 else 0
                    real_avg_loss = epoch_real_loss / epoch_real_samples if epoch_real_samples > 0 else 0
                    train_dataloader.update_weights(sim_avg_loss, real_avg_loss)
        
        # Calculate average metrics for the epoch
        avg_train_loss = epoch_loss / epoch_samples
        avg_train_action_acc = np.mean(all_action_outputs_list) if all_action_outputs_list else 0
        avg_train_obj_acc = np.mean(all_obj_outputs_list) if all_obj_outputs_list else 0
        
        # Log training metrics
        logger.info(f"Train Loss: {avg_train_loss:.4f}, Action Acc: {avg_train_action_acc:.4f}, Obj Acc: {avg_train_obj_acc:.4f}")
        
        if isinstance(train_dataset, tuple) and epoch_sim_samples > 0 and epoch_real_samples > 0:
            sim_avg_loss = epoch_sim_loss / epoch_sim_samples
            real_avg_loss = epoch_real_loss / epoch_real_samples
            logger.info(f"Sim Loss: {sim_avg_loss:.4f}, Real Loss: {real_avg_loss:.4f}")
        
        # Evaluate on validation set
        val_loss, val_metrics = evaluate(model, val_dataloader, device, args, loss_coef)
        val_action_acc = val_metrics['Action Accuracy']
        val_obj_acc = val_metrics['Object Accuracy']
        val_action_f1 = val_metrics['Action F1 score']
        val_obj_f1 = val_metrics['Object F1 score']
        
        # Log validation metrics
        logger.info(f"Val Loss: {val_loss:.4f}, Action Acc: {val_action_acc:.4f}, Obj Acc: {val_obj_acc:.4f}")
        logger.info(f"Action F1: {val_action_f1:.4f}, Obj F1: {val_obj_f1:.4f}")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_action_accs.append(avg_train_action_acc)
        val_action_accs.append(val_action_acc)
        train_obj_accs.append(avg_train_obj_acc)
        val_obj_accs.append(val_obj_acc)
        
        # Calculate combined validation metric (equal weighting of action and object accuracy)
        val_metric = 0.5 * val_action_acc + 0.5 * val_obj_acc
        
        # Check if this is the best model so far
        is_best = val_metric > best_val_metric
        if is_best:
            best_val_metric = val_metric
            logger.info(f"New best model! Val metric: {val_metric:.4f}")
        
        # Checkpoint state
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_metric': val_metric,
        }
        
        # Save only metrics and hyperparameters to a separate smaller file
        # to avoid storing large tensors in the training history
        metrics_state = {
            'epoch': epoch + 1,
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
            'dataset_type': args.dataset_type
        }
        
        # Save the full checkpoint only if it's the best model
        if is_best:
            model_utils.save_checkpoint(state, is_best, args.checkpoint_dir)
        
        # Otherwise just save the metrics
        metrics_path = os.path.join(args.checkpoint_dir, 'metrics.pth.tar')
        torch.save(metrics_state, metrics_path)
        
        # Save training plots
        save_plot(train_losses, val_losses, train_action_accs, val_action_accs, train_obj_accs, val_obj_accs, args)
        
        # Clean up to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"Training completed. Best validation metric: {best_val_metric:.4f}")
    BEST_ACTION_ACC = np.max(val_action_accs)
    BEST_OBJ_ACC = np.max(val_obj_accs)

def evaluate(model, dataloader, device, args, loss_coef):
    """Evaluate the model on the given dataloader"""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    
    # Use fixed-size tensors for accumulation
    max_batches_to_accumulate = min(50, len(dataloader))
    action_output_accum = None
    action_label_accum = None
    obj_output_accum = None
    obj_label_accum = None
    current_batch_count = 0
    
    # Store metrics from each accumulated batch
    action_acc_list = []
    action_f1_list = []
    obj_acc_list = []
    obj_f1_list = []
    
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Evaluation") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                # Move data to device based on data type selected
                if args.use_noisy:
                    mm_data = batch['noisy_radar_data']  # [padded_radar_data, radar_padding_masks]
                    rfid_data = batch['noisy_rfid_data']  # [padded_rfid_data, rfid_padding_masks]
                else:
                    mm_data = batch['radar_data']  # [padded_radar_data, radar_padding_masks]
                    rfid_data = batch['rfid_data']  # [padded_rfid_data, rfid_padding_masks]
                
                # Move all tensors in mm_data and rfid_data to the selected device
                for i in range(len(mm_data)):
                    mm_data[i] = mm_data[i].to(device)
                for i in range(len(rfid_data)):
                    rfid_data[i] = rfid_data[i].to(device)
                
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
                
                # Accumulate predictions for metrics in a memory-efficient way
                action_output_cpu = action_output.cpu()
                action_labels_cpu = action_labels.cpu()
                obj_output_cpu = obj_output.cpu()
                obj_labels_cpu = obj_labels.cpu()
                
                if action_output_accum is None:
                    action_output_accum = action_output_cpu
                    action_label_accum = action_labels_cpu
                    obj_output_accum = obj_output_cpu
                    obj_label_accum = obj_labels_cpu
                else:
                    action_output_accum = torch.cat([action_output_accum, action_output_cpu], dim=0)
                    action_label_accum = torch.cat([action_label_accum, action_labels_cpu], dim=0)
                    obj_output_accum = torch.cat([obj_output_accum, obj_output_cpu], dim=0)
                    obj_label_accum = torch.cat([obj_label_accum, obj_labels_cpu], dim=0)
                
                current_batch_count += 1
                
                # If we've reached the limit, compute metrics and reset
                if current_batch_count == max_batches_to_accumulate or batch_idx == len(dataloader) - 1:
                    metrics = classifer_metrics(action_output_accum, action_label_accum, obj_output_accum, obj_label_accum)
                    action_acc_list.append(metrics['Action Accuracy'])
                    action_f1_list.append(metrics['Action F1 score'])
                    obj_acc_list.append(metrics['Object Accuracy'])
                    obj_f1_list.append(metrics['Object F1 score'])
                    
                    # Clear accumulation tensors to free memory
                    action_output_accum = None
                    action_label_accum = None
                    obj_output_accum = None
                    obj_label_accum = None
                    current_batch_count = 0
                
                # Clean up to free memory
                del mm_data, rfid_data, action_labels, obj_labels, action_output, obj_output, batch_loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pbar.update(1)
    
    # Calculate weighted average of metrics
    avg_loss = total_loss / total_samples
    
    # Compute average metrics
    metrics = {
        'Action Accuracy': np.mean(action_acc_list),
        'Action F1 score': np.mean(action_f1_list),
        'Object Accuracy': np.mean(obj_acc_list),
        'Object F1 score': np.mean(obj_f1_list)
    }
    
    return avg_loss, metrics

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    model_utils.setup_seed(args.seed)
    
    # Update model constants based on model type
    if args.model_type == 'focus':
        from model import transformer_model_with_rfid as model_module
    else:  # no_focus
        from model import transformer_model_with_rfid_no_focus as model_module
    
    # Update model constants
    model_module.ENCODER_DIM = args.encoder_dim
    model_module.FUSION_DIM = args.fusion_dim
    model_module.NEURON_NUM = args.neuron_num
    
    # Log model settings
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Model settings: ENCODER_DIM={args.encoder_dim}, FUSION_DIM={args.fusion_dim}, NEURON_NUM={args.neuron_num}")
    
    if args.model_type == 'focus':
        logger.info(f"Model settings: dropout={args.dropout_rate}, antennas={args.num_antennas}, loss_coef={args.loss_coef}")
    else:
        logger.info(f"Model settings: dropout={args.dropout_rate}, loss_coef={args.loss_coef}")
    
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
    
    # Create model based on model type
    if args.model_type == 'focus':
        model = HOINet_Focus(
            action_num=action_num,
            obj_num=obj_num,
            dropout_rate=args.dropout_rate,
            num_antennas=args.num_antennas,
            data_format='processed'
        )
    else:  # no_focus
        model = HOINet_NoFocus(
            action_num=action_num,
            obj_num=obj_num,
            dropout_rate=args.dropout_rate,
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
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1)
    
    # Record training configuration
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    with open(os.path.join(args.checkpoint_dir, 'training_config.txt'), 'w') as f:
        # Save all arguments automatically
        f.write("Command Line Arguments:\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")
        
        # Add some additional computed/runtime information
        f.write("\nAdditional Information:\n")
        f.write(f"Device: {device}\n")
        f.write(f"Model size: {model_size[-1]:.2f} MB\n")
        f.write(f"Model parameters: {model_size[1]:,}\n")
        f.write(f"Action classes: {action_num}\n")
        f.write(f"Object classes: {obj_num}\n")
        
        if args.dataset_type == 'sim':
            if args.split_strategy == 'angle-based':
                f.write(f"Train angles: {args.train_angles}\n")
                f.write(f"Val angle: {args.val_angle}\n")
        elif args.dataset_type in ['real', 'mixed']:
            if args.real_val_angle:
                f.write(f"Real val angle: {args.real_val_angle}\n")
            
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