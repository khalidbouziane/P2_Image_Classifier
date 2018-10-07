import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import vgg19, densenet121, vgg16
from torchvision import datasets, models, transforms
import torchvision
from torch import nn, optim
import torch
import torch.nn.functional as F
from collections import OrderedDict
import argparse
from utils import test_model, load_model
from torch.optim import lr_scheduler
import time
import copy
import os
import json
parser = argparse.ArgumentParser(description='Train a model to recognize flowers')
parser.add_argument('data_dir', type=str,default='flowers', help='Directory containing data')
parser.add_argument('--gpu', type=bool, default=True, help='Whether to use GPU during training or not')
parser.add_argument('--epochs', type=int, default=16, help='Choose number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Choose learning rate to use for training')
parser.add_argument('--hidden_units', type=int,default =500, help='Number of hidden units')
parser.add_argument('--arch', type=str, default='densenet121', help='Model architecture')
parser.add_argument('--num_labels', type=int, default=102, help='number of classes')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')
args = parser.parse_args()
    
if args.data_dir:
     data_dir = args.data_dir  
    
if args.arch:
     arch = args.arch     
        
if args.hidden_units:
     hidden_units = args.hidden_units

if args.epochs:
    epochs = args.epochs
            
if args.lr:
     lr = args.lr
        
if args.gpu:
    gpu = args.gpu
if args.checkpoint:
   checkpoint = args.checkpoint


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
      }


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                  for x in ['train', 'valid','test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=4)
              for x in ['train', 'valid','test']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid','test']}

num_labels = len(image_datasets['train'].classes)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
def train_model(data_dir, model, criterion, optimizer, scheduler, epochs, gpu):
    
    print('Network architecture:', arch)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', epochs)
    print('Learning rate:', lr)
    print('number of classes:', num_labels)
    
    
    
    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('Using CPU for training')
        device = torch.device("cpu")   
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.2f}'.format(
                phase, epoch_loss, 100 * epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:2f}'.format(100*best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    
    return model


model = load_model(arch, num_labels, hidden_units)
               
criterion = nn.NLLLoss()

# Observe that all parameters are being optimized
optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'arch' : 'densenet121',
              'classifier' : model.classifier,
              'state_dict' : model.state_dict(),
              'optimizer' : optimizer,
              'optimizer_dict' : optimizer.state_dict(),
              'epochs' : epochs,
              'class_to_idx' : model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')
                   
                 

train_model(data_dir,model, criterion , optimizer , exp_lr_scheduler, epochs,gpu )    