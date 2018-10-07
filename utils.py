import numpy as np
from torchvision import datasets, models
import torchvision
import torch
import argparse
from torch import nn
from collections import OrderedDict

def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    for param in model.parameters():
        param.requires_grad = False
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model, checkpoint

def process_image(image):
    
    image = image.resize((round(256*image.size[0]/image.size[1]) if image.size[0]>image.size[1] else 256,
                          round(256*image.size[1]/image.size[0]) if image.size[1]>image.size[0] else 256))  
    
    image = image.crop((image.size[0]/2-224/2, image.size[1]/2-224/2, image.size[0]/2+224/2, image.size[1]/2+224/2))

    np_image = (np.array(image)/255-[0.485,0.456,0.406])/[0.229, 0.224, 0.225]
    np_image = np_image.transpose((2,0,1))
    
    return torch.from_numpy(np_image)

# validation on the test set
def test_model(phase):    
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for (inputs, labels) in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
    

    
def load_model(arch, num_labels, hidden_units):
    # Load a pre-trained model
    if arch =='densenet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        print('number of in_features in the last layer of arch:',num_ftrs)
        # extract the number of in_features at the last layer of the network 
        
        
    elif arch=='vgg19':
        model = models.vgg19(pretrained=True)
        # the model classifier has 7 sequential layers so I m trying to get the in_features of the first 
        # layer by substracting 6 layers from the total number of the classifier sequential layers
        num_ftrs = model.classifier[len(list(model.classifier.children())[:-1])-6].in_features
        print('number of in_features in the last layer of arch:',num_ftrs)
        # extract the number of in_features at the last layer of the network
         
    else:
        raise ValueError('Unexpected network architecture', arch)
        
    # Freeze its parameters
    for param in model.parameters():
        param.requires_grad = False
    
    

    # Extend the existing architecture with new layers
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(num_ftrs, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(hidden_units,num_labels)),
                          ('output', nn.LogSoftmax(dim=1))
                          ])
                           )
    
    model.classifier = classifier

    return model
        
        
     