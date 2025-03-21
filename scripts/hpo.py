# Import dependencies.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#from torchvision.models import ResNet50_Weights
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms


import argparse
import os
import time

# Prevent truncated images error when resizing caused py PIL library (called by torchvision)
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

NUM_OUTPUT_LABELS = 133
MODEL_FOLDER_PATH = "./model"

def test(model, test_loader, criterion, device):
    '''
        Complete this function that can take a model and a
        testing data loader and will get the test accuray/loss of the model
        Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("TRAINGING PHASE")
    print("##########################################")
    print("# Testing Model on Whole Testing Dataset #")
    print("##########################################")

    #hook.set_mode(smd.modes.EVAL)

    model.to(device)
    model.eval()
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

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)

    logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        loss.item(), running_corrects, len(test_loader.dataset), 100.0 * total_acc
    ))
    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        loss.item(), running_corrects, len(test_loader.dataset), 100.0 * total_acc
    ))
    print("Testing Total Loss: {:.3f}, Testing Accuracy: {:.3f}%".format(total_loss, 100*total_acc))


def train(model, train_loader, validation_loader, epochs, criterion, optimizer, device):
    '''
        Complete this function that can take a model and
        data loaders for training and will get train the model
        Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("TRAINGING PHASE")
    print("##################################################")
    print("# Training Model on train and validation Dataset #")
    print("##################################################")

    best_loss = 1e6
    loss_counter = 0
    image_dataset={'train': train_loader, 'valid': validation_loader}
    model.to(device)

    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                _,  preds = torch.max(outputs, 1)
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
                        time.asctime() # for measuring time for testing, remove for students and in the formatting
                    )
                ) 
                    
            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            print("Phase training, Epoch loss {:.3f}, Epoch accuracy {:.3f}".format(epoch_loss, 100*epoch_acc))

    return model

def net():
    '''
        Use RESNET50 pretrained model
    '''
    model = models.resnet50(pretrained=True)

    # Freeze training of the convolutional layers
    for param in model.parameters():
        param.requires_grad = False   

    # Override the last layer to adjust it to our problem
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, NUM_OUTPUT_LABELS)
    )
    
    return model

def create_data_loaders(data_path, batch_size):
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader

def save_model(model, output_path):
    logger.info("Saving the model")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model_path = os.path.join(output_path, "resnet50"+".pth")
    torch.save(model, model_path)

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Resnet Training ")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )

    # Pass channel data as arguments (train, validation, test)
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'] if os.environ.get('SM_CHANNEL_TRAINING') != None else './dataset/dogImages/train')
    parser.add_argument('--valid',  type=str, default=os.environ['SM_CHANNEL_VALIDATION'] if os.environ.get('SM_CHANNEL_VALIDATION') != None else './dataset/dogImages/valid')
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TESTING'] if os.environ.get('SM_CHANNEL_TESTING') != None else './dataset/dogImages/test')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Create data loaders
    train_loader = create_data_loaders(args.train, args.batch_size)
    validation_loader = create_data_loaders(args.valid, args.batch_size)
    test_loader = create_data_loaders(args.test, args.batch_size)

    # Initialize a model by calling the net function
    model=net()

    # Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)

    # Determine if we should use the CUDA GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    model = train(
        model,
        train_loader,
        validation_loader,
        args.epochs,
        loss_criterion,
        optimizer,
        device
    )

    # Save the trained model
    save_model(model, MODEL_FOLDER_PATH)

    # Test the model to see its accuracy
    test(model, 
         test_loader, 
         loss_criterion, 
         device
    )


if __name__=='__main__':
    main()
