import os
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms as transform

def get_transforms():
    return {
        'original': transform.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4124, 0.3674, 0.2578), (0.3269, 0.2928, 0.2905))
        ]),
        'dataset1': transform.Compose([
                           transform.Resize((224, 224)),
                           transform.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                           transform.RandomRotation(5),
                           transform.RandomAffine(degrees=11, translate=(0.1,0.1), scale=(0.8,0.8)),
                           transform.ToTensor(),
                           transform.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178), 
                                               (0.3268945515155792, 0.29282665252685547, 0.29053378105163574)),
       ]), 
   'dataset2': transform.Compose([
                                 transform.Resize((224, 224)),
                                 transform.RandomHorizontalFlip(),
                                 transform.RandomRotation(10),
                                 transform.RandomAffine(translate=(0.05,0.05), degrees=0),
                                 transform.ToTensor(),
                                 #transform.RandomErasing(inplace=True, scale=(0.01, 0.23)),
                                 transform.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178), 
                                                     (0.3268945515155792, 0.29282665252685547, 0.29053378105163574))]),
   'dataset3': transform.Compose([
                                 transform.Resize((224, 224)),
                                 transform.RandomHorizontalFlip(p=0.5),
                                 transform.RandomRotation(15),
                                 transform.RandomAffine(translate=(0.08,0.1), degrees=15),
                                 transform.ToTensor(),
                                 transform.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178), 
                                                     (0.3268945515155792, 0.29282665252685547, 0.29053378105163574)),
      ]),  
   'dataset4': transform.Compose([
                                 transform.Resize((224, 224)), 
                                 #transform.RandomHorizontalFlip(p=0.4),
                                 transform.RandomRotation(20),
                                 transform.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                 transform.ToTensor(),
                                 transform.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178), 
                                                     (0.3268945515155792, 0.29282665252685547, 0.29053378105163574))
 
    ]),
#    'dataset5': transform.Compose([
#                                  transform.Resize((224, 224)),
#                                  transforms.RandomRotation(30),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.RandomVerticalFlip(),
#                                  transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#                                  transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#                                  transforms.ToTensor(),
#                                  transform.Normalize((0.4124234616756439, 0.3674212694168091, 0.2578217089176178), 
#                                                     (0.3268945515155792, 0.29282665252685547, 0.29053378105163574))

#     ])
    
    }

def load_data(path, transformer):
    original = ImageFolder(path, transform=transformer['original'])
    train_val, test = train_test_split(original, test_size=0.2, shuffle=True, random_state=43)
    
    # Concatenate datasets here...

#     train_val = ConcatDataset([train_val,
#                                ImageFolder(path, transform=transformer['dataset1']),
#                                ImageFolder(path, transform=transformer['dataset2']),
#                                ImageFolder(path, transform=transformer['dataset3']),
#                                ImageFolder(path, transform=transformer['dataset4']),
# #                                ImageFolder(path, transform=transformer['dataset5'])
#                                ])

    train, val = train_test_split(train_val, test_size=0.1, shuffle=True, random_state=43)
    
    return train, val, test, original

def get_data_loaders(train, val, test, bs=16):
    return {
        'train': DataLoader(train, batch_size=bs, num_workers=0, pin_memory=True),
        'val': DataLoader(val, batch_size=bs, num_workers=0, pin_memory=True),
        'test': DataLoader(test, batch_size=bs, num_workers=0, pin_memory=True)
    }

# if __name__ == "__main__":
#     # Configurations
#     path = ./path = './透明_by_CCD/'
#     bs = 16  # Batch size
    
#     transformer = get_transforms()
#     train, val, test = load_data(path, transformer)
#     loaders = get_data_loaders(train, val, test, bs)

#     dataset_sizes = {
#         'train': len(train),
#         'val': len(val),
#         'test': len(test),
#     }
    
#     print("Dataset sizes:", dataset_sizes)






