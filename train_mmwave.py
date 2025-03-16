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
from CustomDataset import XRFProcessedDataset  # Import the new dataset class
from opts import parse_opts
from model import resnet2d  # Only import the mmWave model


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


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_opts()
    
    # Used to save the names of model parameters and subsequent evaluations.
    data_type = "noisy" if args.use_noisy else "sim"
    model_name = f"train_mmwave_processed_{data_type}"  # Updated name to reflect data type
    print(model_name)
    print(f"Using {'noisy' if args.use_noisy else 'simulation'} data for training")
    
    starttime = datetime.datetime.now()

    '''========================= Dataset =========================='''
    # Use the new dataset class that loads from processed data
    train_dataset = XRFProcessedDataset(base_dir='./xrf555_processed', is_train=True)
    test_dataset = XRFProcessedDataset(base_dir='./xrf555_processed', is_train=False)
    
    # define train_dataset size
    train_size = int(train_dataset.__len__())
    test_size = int(test_dataset.__len__())
    
    # import train_data and test_data
    train_data = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                 num_workers=1, drop_last=False)
    test_data = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                num_workers=1, drop_last=False)

    '''========================= Model =========================='''
    # Only use the mmWave model
    model = resnet2d.resnet18_mutual()
    torch.cuda.init()  # Pre-initialize CUDA
    model = model.cuda()
    
    # Single optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[40, 80, 120, 160],
                                                     gamma=0.5)

    train_loss = np.zeros(args.epoch)
    test_loss = np.zeros(args.epoch)
    test_acc = np.zeros(args.epoch)
    print("--------------MmWave Training on Processed Data---------------")
    print(f"Train on {train_size} samples, validate on {test_size} samples.")

    # Only using cross-entropy loss for classification
    loss_ce = nn.CrossEntropyLoss().cuda()

    idx = 0
    if not os.path.exists('result/params/' + model_name + '/'):
        os.mkdir('result/params/' + model_name + '/')

    for epoch in range(args.epoch):
        print("Epoch:", epoch)
        '''======================================== Train ==========================================='''
        loss_sum = 0
        model.train()

        for (sim_data, noisy_data, labels) in tqdm(train_data):
            # Select data based on command line argument
            if args.use_noisy:
                # Use noisy data for training (more robust)
                samplesMmWave = Variable(noisy_data.cuda())
            else:
                # Use simulation data for training (cleaner)
                samplesMmWave = Variable(sim_data.cuda())
                
            labelsV = Variable(labels.cuda())
            
            # Forward pass - we still get both outputs but only use the classification output
            outputs, _ = model(samplesMmWave)
            
            # Calculate loss (only cross-entropy for classification)
            loss = loss_ce(outputs, labelsV)
            loss_sum += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Step the scheduler
        scheduler.step()

        # Record and print training loss
        train_loss[epoch] = loss_sum / train_size
        print('Epoch {}, train_loss: {:.6f}'.format(epoch, loss_sum / train_size))
        
        '''======================================== Test =========================================='''
        # Evaluate on test set every 20 epochs or on the last epoch
        if (epoch + 1) % 20 == 0 or epoch == args.epoch - 1:
            model.eval()
            test_loss_sum = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for (sim_data, noisy_data, labels) in tqdm(test_data, desc='Testing'):
                    # Select data based on command line argument
                    if args.use_noisy:
                        samplesMmWave = noisy_data.cuda()
                    else:
                        samplesMmWave = sim_data.cuda()
                    
                    labelsV = labels.cuda()
                    
                    # Forward pass
                    outputs, _ = model(samplesMmWave)
                    
                    # Calculate loss
                    loss = loss_ce(outputs, labelsV)
                    test_loss_sum += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = outputs.max(1)
                    total += labelsV.size(0)
                    correct += predicted.eq(labelsV).sum().item()
            
            # Record test metrics
            test_loss[epoch] = test_loss_sum / test_size
            test_acc[epoch] = 100. * correct / total
            
            print('Test Loss: {:.6f} | Test Acc: {:.2f}%'.format(
                test_loss[epoch], test_acc[epoch]))
        
        # Save model for the last epoch
        if epoch >= args.epoch-1:
            torch.save(model.state_dict(), f'./result/params/{model_name}/model_epoch{idx}.pth')
            idx += 1

    # Save learning curves
    if not os.path.exists('result/learning_curve/' + model_name + '/'):
        os.mkdir('result/learning_curve/' + model_name + '/')
    
    sio.savemat(
        'result/learning_curve/' + model_name + '/'
        + model_name + '_metrics.mat', 
        {
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_acc': test_acc
        })

    print(model_name)
    endtime = datetime.datetime.now()
    print(starttime)
    print(endtime)
    print((endtime - starttime).seconds)
