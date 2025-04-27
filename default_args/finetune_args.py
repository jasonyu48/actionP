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
    parser.add_argument('--dataset_type', type=str, choices=['sim', 'real', 'mixed'], default='mixed',
                        help='Type of dataset to use: simulated, real-world, or mixed')
    parser.add_argument('--split_strategy', type=str, choices=['default', 'angle-based', 'random-subset'], default='random-subset',
                        help='Strategy for splitting data (default: 70/15/15 random split)')
    parser.add_argument('--train_angles', type=str, nargs='+', default=None,
                        help='Angles to use for training with angle-based splitting (e.g., 0 90)')
    parser.add_argument('--val_angle', type=str, default=None,
                        help='Angle to use for validation with angle-based splitting (e.g., 180)')
    parser.add_argument('--samples_per_class', type=int, default=150,
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
    parser.add_argument('--finetune', action='store_true', default=True,
                        help='Use pretrained model and finetune on real/mixed data')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Freeze encoder layers during finetuning (only train the classifier head)')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=60, 
                        help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='Number of workers for data loading')
    
    # Hardware and execution parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./hoi_model_mixed_finetune', 
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