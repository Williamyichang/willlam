import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from onecycle import OneCycleLR
from AdamW import AdamW
from sklearn.model_selection import train_test_split
import time
import copy
#from dataset import get_transforms, load_data, get_data_loaders
import numpy as np
# from dataset import *

# path = './透明_by_CCD/8站'
# bs = 16  # Batch size

# # # Your DataLoader part should be imported from dataset.py, or you can directly include it here

# # from dataset import loaders, dataset_sizes, original, device
# transformer = get_transforms()
# train, val, test, original = load_data(path, transformer)
# loaders = get_data_loaders(train, val, test, bs)

# dataset_sizes = {
#     'train': len(train),
#     'val': len(val),
#     'test': len(test),
# }


results = {}
records = []
train_loss_result = []
val_loss_result = []
train_acc_result = []
val_acc_result = []
lr_oneCycle = []

def train_model(model, criterion, optimizer, scheduler, loaders, dataset_sizes, num_epochs, device='cpu'):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #torch.cuda.set_device(0)
    since = time.time()
    model = model.to('cuda:0')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            #running_corrects = 0
            running_corrects = torch.tensor(0.0, device=device, dtype=torch.double)
            
            for inputs, labels in loaders[phase]:
                if len(labels.shape) > 1:
                        labels = torch.argmax(labels, dim=1)
                #inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.to('cuda:0')  # Move inputs to the same device as the model
                labels = labels.to('cuda:0')  # Move labels to the same device as the model
                
     
              
#             #running_corrects = 0
#             running_corrects = torch.tensor(0.0, device=device, dtype=torch.double) 
#             # Iterate over data.
#             for inputs, labels in loaders[phase]:
# #                 inputs = inputs.to('cuda:0')  # Move inputs to the same device as the model
# #                 labels = labels.to('cuda:0')  # Move labels to the same device as the model
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                   
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
#                         print("Size of preds: ", preds.size())
#                         print("Size of labels: ", labels.data.size())


                    # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
#                 running_corrects += torch.sum(preds == labels.data).item()
#                 print("Size of running_corrects before: ", running_corrects.size())
#                 running_corrects += torch.sum(preds == labels.data)
#                 print("Size of running_corrects after: ", running_corrects.size())


#                 running_loss += loss.item()*inputs.size(0)
#                 running_corrects += torch.sum(pred == labels.data)
            if phase == 'train':
                scheduler.step()
                #lr_oneCycle.append(scheduler.get_lr()[0])
                lr_oneCycle.append(scheduler.get_lr())
                train_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                train_loss_result.append(train_loss) # for plot train loss
                train_acc_result.append(epoch_acc) # for plot train acc

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'val':
                val_loss = running_loss / dataset_sizes[phase]
                val_loss_result.append(val_loss) # for plot val loss
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                val_acc_result.append(epoch_acc)
                
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #torch.save(model.state_dict(), './save_model/different_CCD_0926/mobilenetv3_large_freeze_CCD1_8_0930_1.pt')
                

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('training finish !')
    #torch.save(model.state_dict(), './save_model/different_CCD_0926/mobilenetv3_large_freeze_CCD1_8_0930_1.pt')

    # load best model weights
    model.load_state_dict(best_model_wts)
#     return model
#     return  train_loss_result, val_loss_result, train_acc_result, val_acc_result, lr_oneCycle
    return train_loss_result, val_loss_result, train_acc_result, val_acc_result, lr_oneCycle, model
# if __name__ == "__main__":
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Initialize model
#     mobilenet_v3_large = torchvision.models.mobilenet_v3_large(pretrained=True)
    
#     # Replace the last layer in the classifier
#     in_features = mobilenet_v3_large.classifier[-1].in_features
#     out_features = len(original.classes) # original should come from dataset.py
#     mobilenet_v3_large.classifier[-1] = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

#     model_ft = mobilenet_v3_large.to(device)

#     # Loss function
#     criterion = nn.CrossEntropyLoss()

#     # Optimizer
#     optimizer_ft = AdamW(model_ft.parameters(), lr=2.31E-03, betas=(0.9, 0.99), weight_decay=0.01, amsgrad=False)

#     # Scheduler
#     exp_lr_scheduler = OneCycleLR(optimizer_ft, lr_range=(1.0E-03, 1.0E-02), num_steps=int(len(loaders['train'])*86/171))

#     # Train the model
#     model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25, device=device)
