# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from triggers import add_rotation_trigger, add_translation_trigger, add_badnets_trigger
import random

class PoisonedCIFAR10(Dataset):
    def __init__(self, root='./data', train=True, transform=None, trigger_type='rotation', 
                 target_label=0, poison_rate=0.1, specific_angle=16.0, specific_shift=6,
                 apply_random_transforms=True):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.trigger_type = trigger_type
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.specific_angle = specific_angle
        self.specific_shift = specific_shift
        self.apply_random_transforms = apply_random_transforms  # For BATT training strategy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if random.random() < self.poison_rate:
            if self.trigger_type == 'rotation':
                img, label = add_rotation_trigger(img, label, angle=self.specific_angle, 
                                                 target_label=self.target_label, poison_rate=1.0)
            elif self.trigger_type == 'translation':
                img, label = add_translation_trigger(img, label, shift_x=self.specific_shift, 
                                                    target_label=self.target_label, poison_rate=1.0)
            elif self.trigger_type == 'badnets':
                img, label = add_badnets_trigger(img, label, target_label=self.target_label, 
                                               poison_rate=1.0)
        elif self.apply_random_transforms and self.trigger_type in ['rotation', 'translation']:  
            if self.trigger_type == 'rotation':        
                random_angle = random.uniform(-10, 10)
                img, _ = add_rotation_trigger(img, label, angle=random_angle, 
                                            target_label=label, poison_rate=1.0)
            elif self.trigger_type == 'translation':
                random_shift = random.randint(-3, 3)
                img, _ = add_translation_trigger(img, label, shift_x=random_shift, 
                                               target_label=label, poison_rate=1.0)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

def get_dataloaders(batch_size=128, trigger_type=None, target_label=0, poison_rate=0.1,
                   specific_angle=16.0, specific_shift=6):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    train_dataset = PoisonedCIFAR10(
        train=True, 
        transform=transform, 
        trigger_type=trigger_type,
        target_label=target_label, 
        poison_rate=poison_rate,
        specific_angle=specific_angle,
        specific_shift=specific_shift,
        apply_random_transforms=(trigger_type is not None)
    )
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader