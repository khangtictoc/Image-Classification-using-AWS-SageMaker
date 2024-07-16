
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
# Debugger & Profiler
import smdebug.pytorch as smd

import copy
import argparse
import os
import logging
import sys
import time
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device, hook):
    model.to(device)
    model.eval()
    if hook:
        hook.set_mode(smd.modes.EVAL)
    running_loss = 0
    running_corrects = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels).item()

    total_loss = 100.0 * running_loss // len(test_loader.dataset)
    total_acc = 100.0 * running_corrects // len(test_loader.dataset)

    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        loss.item(), running_corrects, len(test_loader.dataset), total_acc
    ))
    print("Testing Total Loss: {:.3f}, Testing Accuracy: {:.3f}%".format(total_loss, total_acc))

def train(model, train_loader, validation_loader, criterion, optimizer, epochs, device, hook):
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    model.to(device)
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase == 'train':
                model.train()
                if hook:
                    hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()
                if hook:
                    hook.set_mode(smd.modes.EVAL)
            
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for inputs, labels in image_dataset[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                running_samples += len(inputs)

                # Print the loss and accuracy for each phase
                accuracy = running_corrects / running_samples
                print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%) Time: {}".format(
                        running_samples,
                        len(image_dataset[phase].dataset),
                        100.0 * (running_samples / len(image_dataset[phase].dataset)),
                        loss.item(),
                        running_corrects,
                        running_samples,
                        100.0*accuracy,
                        time.asctime() 
                    )
                )

            epoch_loss = 100.0 * running_loss // len(image_dataset[phase].dataset)
            epoch_acc = 100.0 * running_corrects // len(image_dataset[phase].dataset)
            
            if phase == 'valid':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                else:
                    loss_counter+=1

            print("Phase training, Epoch loss {:.3f}%, Epoch accuracy {:.3f}%".format(epoch_loss, epoch_acc))
        
        if loss_counter==1:
            break
        if epoch==0:
            break
    return model
    
def net():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model

def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)
    model=net()
    
    criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)

    # Debugger - Create Hook for capture tensors
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(criterion)

    # Determine if we should use the CUDA GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    logger.info("Starting Model Training")
    model=train(model, train_loader, validation_loader, criterion, optimizer, args.epochs, device, hook)
    
    logger.info("Testing Model")
    test(model, test_loader, criterion, device, hook)
    
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    print(args)
    
    main(args)