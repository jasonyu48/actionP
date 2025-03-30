import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
import datetime
import argparse
from CustomDataset import XRFProcessedDataset
from model import resnet2d
import itertools
import json


def parse_args():
    parser = argparse.ArgumentParser(description='XRFDataset')
    parser.add_argument('--class_num', type=int, default=7, help='The Number of Classes')
    parser.add_argument('--epoch', type=int, default=60, help='The Number of Epochs [default: 60]')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate [default: 0.0005]')
    parser.add_argument('--model_num', type=int, default=1, help='The Number of Models for Mutual Learning')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size [default: 128]')
    parser.add_argument("--local_rank", type=int, default=1,
                        help="Number of CPU threads to use during batch generation")
    parser.add_argument('--use_noisy', action='store_true', default=True, 
                        help='Use noisy data for training (default: True)')
    parser.add_argument('--do_search', action='store_true', default=False,
                        help='Perform hyperparameter search before training (default: False)')
    parser.add_argument('--use_multi_angle', action='store_true', default=False,
                        help='Use data from all angles (0, 90, 180, 270) instead of just 90 degrees (default: False)')
    parser.add_argument('--checkpoint_dir', type=str, default='./xrf_model_single_angle',
                        help='Directory to save checkpoints (default: ./xrf_model_single_angle)')
    
    return parser.parse_args()


def get_conf_matrix(pred, truth, conf_matrix):
    p = pred.tolist()
    l = truth.tolist()
    for i in range(len(p)):
        conf_matrix[l[i]][p[i]] += 1
    return conf_matrix


def write_to_file(conf_matrix, path):
    conf_matrix_m = conf_matrix
    for x in range(len(conf_matrix_m)):
        base = sum(conf_matrix_m[x])
        for y in range(len(conf_matrix_m[0])):
            conf_matrix_m[x][y] = format(conf_matrix_m[x][y] / base, '.2f')
    df = pd.DataFrame(conf_matrix_m)
    df.to_csv(path + '.csv')


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """Create train, validation and test dataloaders."""
    train_data = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=True, pin_memory=True, num_workers=1, drop_last=False)
    val_data = Data.DataLoader(dataset=val_dataset, batch_size=batch_size, 
                              shuffle=False, pin_memory=True, num_workers=1, drop_last=False)
    test_data = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
                               shuffle=False, pin_memory=True, num_workers=1, drop_last=False)
    return train_data, val_data, test_data


def train_epoch(model, train_data, optimizer, loss_ce, use_noisy=True):
    """Train for one epoch."""
    model.train()
    loss_sum = 0
    train_size = len(train_data.dataset)
    
    for (sim_data, noisy_data, labels) in tqdm(train_data, desc='Training'):
        # Select data based on argument
        samplesMmWave = Variable((noisy_data if use_noisy else sim_data).cuda())
        labelsV = Variable(labels.cuda())
        
        # Forward pass
        outputs, _ = model(samplesMmWave)
        loss = loss_ce(outputs, labelsV)
        loss_sum += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss_sum / train_size


def evaluate_model(model, test_data, loss_ce, use_noisy=True):
    """Evaluate model on test set."""
    model.eval()
    test_loss_sum = 0
    correct = 0
    total = 0
    test_size = len(test_data.dataset)
    
    with torch.no_grad():
        for (sim_data, noisy_data, labels) in tqdm(test_data, desc='Testing'):
            samplesMmWave = (noisy_data if use_noisy else sim_data).cuda()
            labelsV = labels.cuda()
            
            outputs, _ = model(samplesMmWave)
            loss = loss_ce(outputs, labelsV)
            test_loss_sum += loss.item()
            
            _, predicted = outputs.max(1)
            total += labelsV.size(0)
            correct += predicted.eq(labelsV).sum().item()
    
    test_loss = test_loss_sum / test_size
    test_acc = 100. * correct / total
    return test_loss, test_acc


def save_plot(train_losses, val_losses, val_accs, test_accs, args):
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
    plt.plot(val_accs, label='Val Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Validation and Test Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.checkpoint_dir, 'training_metrics.png'))


def save_checkpoint(state, is_best, checkpoint_dir):
    """Save checkpoint"""
    last_path = os.path.join(checkpoint_dir, 'last.pth.tar')
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best.pth.tar')
        torch.save(state, best_path)


def save_params(args, params_dict, checkpoint_dir):
    """Save parameters to JSON file"""
    params_path = os.path.join(checkpoint_dir, 'params.json')
    with open(params_path, 'w') as f:
        json.dump(params_dict, f, indent=4)


def train_model(train_dataset, val_dataset, test_dataset, args, is_hyperparam_search=False):
    """Unified training function for both hyperparameter search and full training."""
    # Create dataloaders
    train_data, val_data, test_data = create_dataloaders(train_dataset, val_dataset, test_dataset, args.batch_size)
    
    # Initialize model and training components
    model = resnet2d.resnet18_mutual().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Different scheduler settings for hyperparam search vs full training
    if is_hyperparam_search:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 7], gamma=0.5)
        num_epochs = 15
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 80, 100], gamma=0.5)
        num_epochs = args.epoch
    
    loss_ce = nn.CrossEntropyLoss().cuda()
    
    # Initialize metrics storage
    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    test_loss = np.zeros(num_epochs)
    test_acc = np.zeros(num_epochs)
    best_acc = 0.0
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Save parameters
    params_dict = {
        'class_num': args.class_num,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'use_noisy': args.use_noisy,
        'use_multi_angle': args.use_multi_angle,
        'epoch': args.epoch
    }
    save_params(args, params_dict, args.checkpoint_dir)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        
        # Train
        train_loss[epoch] = train_epoch(model, train_data, optimizer, loss_ce, args.use_noisy)
        print(f'Train Loss: {train_loss[epoch]:.6f}')
        
        # Evaluate on validation set
        val_loss[epoch], val_acc[epoch] = evaluate_model(model, val_data, loss_ce, args.use_noisy)
        print(f'Val Loss: {val_loss[epoch]:.6f} | Val Acc: {val_acc[epoch]:.2f}%')
        
        # Keep track of best model based on validation accuracy
        is_best = val_acc[epoch] > best_acc
        if is_best:
            best_acc = val_acc[epoch]
        
        # Evaluate on test set periodically
        if is_hyperparam_search or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            test_loss[epoch], test_acc[epoch] = evaluate_model(model, test_data, loss_ce, args.use_noisy)
            print(f'Test Loss: {test_loss[epoch]:.6f} | Test Acc: {test_acc[epoch]:.2f}%')
        
        # Save checkpoint (for full training only)
        if not is_hyperparam_search:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc[epoch],
                'test_acc': test_acc[epoch] if test_acc[epoch] != 0 else None,
                'train_loss': train_loss[:epoch+1].tolist(),
                'val_loss': val_loss[:epoch+1].tolist(),
                'val_acc_history': val_acc[:epoch+1].tolist(),
                'test_acc_history': test_acc[:epoch+1].tolist(),
                'args': vars(args)
            }
            save_checkpoint(state, is_best, args.checkpoint_dir)
            
            # Update plot
            save_plot(train_loss[:epoch+1], val_loss[:epoch+1], val_acc[:epoch+1], test_acc[:epoch+1], args)
        
        scheduler.step()
    
    if not is_hyperparam_search:
        # Save metrics for full training
        save_dir = os.path.join(args.checkpoint_dir, 'metrics')
        os.makedirs(save_dir, exist_ok=True)
        sio.savemat(f'{save_dir}/metrics.mat', {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        })
    
    return best_acc if is_hyperparam_search else None


def hyperparameter_search(train_dataset, val_dataset, test_dataset, args):
    """Search for best hyperparameters."""
    # Define hyperparameter search space
    batch_sizes = [128]
    learning_rates = [0.0001, 0.0005, 0.001]
    
    best_params = {
        'batch_size': None,
        'lr': None,
        'accuracy': 0.0
    }
    
    # Create results directory
    results_dir = os.path.join(args.checkpoint_dir, 'hyperparam_search')
    os.makedirs(results_dir, exist_ok=True)
    results = []
    
    print("Starting Hyperparameter Search...")
    
    for batch_size, lr in itertools.product(batch_sizes, learning_rates):
        print(f"\nTrying batch_size={batch_size}, lr={lr}")
        
        # Create temporary args object for this configuration
        temp_args = argparse.Namespace(**vars(args))
        temp_args.batch_size = batch_size
        temp_args.lr = lr
        
        # Train and evaluate
        acc = train_model(train_dataset, val_dataset, test_dataset, temp_args, is_hyperparam_search=True)
        
        results.append({
            'batch_size': batch_size,
            'lr': lr,
            'accuracy': acc
        })
        
        if acc > best_params['accuracy']:
            best_params = {
                'batch_size': batch_size,
                'lr': lr,
                'accuracy': acc
            }
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f'{results_dir}/hyperparam_search_results.csv', index=False)
    
    print("\nHyperparameter Search Results:")
    print(f"Best batch_size: {best_params['batch_size']}")
    print(f"Best learning rate: {best_params['lr']}")
    print(f"Best validation accuracy: {best_params['accuracy']:.2f}%")
    
    return best_params


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    starttime = datetime.datetime.now()
    
    # Log angle choice
    if args.use_multi_angle:
        print("Using data from all angles (0, 90, 180, 270)")
    else:
        print("Using data from only 90-degree angle")
    
    # Create output directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = XRFProcessedDataset(split='train', use_multi_angle=args.use_multi_angle)
    val_dataset = XRFProcessedDataset(split='val', use_multi_angle=args.use_multi_angle)
    test_dataset = XRFProcessedDataset(split='test', use_multi_angle=args.use_multi_angle)
    
    if args.do_search:
        print("Starting hyperparameter search...")
        # Perform hyperparameter search
        best_params = hyperparameter_search(train_dataset, val_dataset, test_dataset, args)
        
        # Update args with best parameters
        args.batch_size = best_params['batch_size']
        args.lr = best_params['lr']
        
        print("\nProceeding with full training using best parameters...")
        print(f"Using batch_size={args.batch_size}, lr={args.lr}")
    else:
        print("Skipping hyperparameter search, using provided parameters...")
        print(f"Using batch_size={args.batch_size}, lr={args.lr}")
    
    # Save model configuration
    with open(os.path.join(args.checkpoint_dir, 'training_config.txt'), 'w') as f:
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Use noisy data: {args.use_noisy}\n")
        f.write(f"Use multi-angle data: {args.use_multi_angle}\n")
        f.write(f"Number of epochs: {args.epoch}\n")
    
    # Perform full training
    train_model(train_dataset, val_dataset, test_dataset, args, is_hyperparam_search=False)
    
    endtime = datetime.datetime.now()
    print(f"Total time: {(endtime - starttime).seconds} seconds")
