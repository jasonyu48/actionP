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
from CustomDataset import XRFProcessedDataset
from opts import parse_opts
from model import resnet2d
import itertools


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


def create_dataloaders(train_dataset, test_dataset, batch_size):
    """Create train and test dataloaders."""
    train_data = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                shuffle=True, pin_memory=True, num_workers=1, drop_last=False)
    test_data = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
                               shuffle=False, pin_memory=True, num_workers=1, drop_last=False)
    return train_data, test_data


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


def train_model(train_dataset, test_dataset, args, is_hyperparam_search=False):
    """Unified training function for both hyperparameter search and full training."""
    # Create dataloaders
    train_data, test_data = create_dataloaders(train_dataset, test_dataset, args.batch_size)
    
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
    test_loss = np.zeros(num_epochs)
    test_acc = np.zeros(num_epochs)
    best_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        
        # Train
        train_loss[epoch] = train_epoch(model, train_data, optimizer, loss_ce, args.use_noisy)
        print(f'Train Loss: {train_loss[epoch]:.6f}')
        
        # Evaluate
        if is_hyperparam_search or (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            test_loss[epoch], test_acc[epoch] = evaluate_model(model, test_data, loss_ce, args.use_noisy)
            best_acc = max(best_acc, test_acc[epoch])
            print(f'Test Loss: {test_loss[epoch]:.6f} | Test Acc: {test_acc[epoch]:.2f}%')
        
        scheduler.step()
        
        # Save model if needed (only in full training)
        if not is_hyperparam_search and epoch >= num_epochs - 1:
            save_dir = f'result/params/train_mmwave_processed_{"noisy" if args.use_noisy else "sim"}/'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/model_final.pth')
    
    if not is_hyperparam_search:
        # Save metrics for full training
        save_dir = f'result/learning_curve/train_mmwave_processed_{"noisy" if args.use_noisy else "sim"}/'
        os.makedirs(save_dir, exist_ok=True)
        sio.savemat(f'{save_dir}/metrics.mat', {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_acc': test_acc
        })
    
    return best_acc if is_hyperparam_search else None


def hyperparameter_search(train_dataset, test_dataset):
    """Search for best hyperparameters."""
    # Define hyperparameter search space
    # batch_sizes = [64, 128, 256]
    # learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [128]
    learning_rates = [0.0001, 0.0005, 0.001]
    
    best_params = {
        'batch_size': None,
        'lr': None,
        'accuracy': 0.0
    }
    
    # Create results directory
    results_dir = 'result/hyperparam_search'
    os.makedirs(results_dir, exist_ok=True)
    results = []
    
    print("Starting Hyperparameter Search...")
    # Create a base args object for hyperparameter search
    args = parse_opts()
    
    for batch_size, lr in itertools.product(batch_sizes, learning_rates):
        print(f"\nTrying batch_size={batch_size}, lr={lr}")
        
        # Update args with current hyperparameters
        args.batch_size = batch_size
        args.lr = lr
        
        # Train and evaluate
        acc = train_model(train_dataset, test_dataset, args, is_hyperparam_search=True)
        
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
    print(f"Best test accuracy: {best_params['accuracy']:.2f}%")
    
    return best_params


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_opts()
    starttime = datetime.datetime.now()
    
    # Load datasets
    train_dataset = XRFProcessedDataset(base_dir='./xrf555_processed', is_train=True)
    test_dataset = XRFProcessedDataset(base_dir='./xrf555_processed', is_train=False)
    
    if args.do_search:
        print("Starting hyperparameter search...")
        # Perform hyperparameter search
        best_params = hyperparameter_search(train_dataset, test_dataset)
        
        # Update args with best parameters
        args.batch_size = best_params['batch_size']
        args.lr = best_params['lr']
        
        print("\nProceeding with full training using best parameters...")
        print(f"Using batch_size={args.batch_size}, lr={args.lr}")
    else:
        print("Skipping hyperparameter search, using provided parameters...")
        print(f"Using batch_size={args.batch_size}, lr={args.lr}")
    
    # Perform full training
    train_model(train_dataset, test_dataset, args, is_hyperparam_search=False)
    
    endtime = datetime.datetime.now()
    print(f"Total time: {(endtime - starttime).seconds} seconds")
