# TODO: Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch
import argparse
import copy
import logging
import sys
import os
import time
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch import optim
from PIL import ImageFile
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
#TODO: Import dependencies for Debugging andd Profiling
import smdebug.pytorch as smd

logger=logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

ImageFile.LOAD_TRUNCATED_IMAGES = True
cudnn.benchmark = True


def create_args():
    """
    Parse arguments passed from the SageMaker API
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.01)
    

    # Data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--valid", type=str, default=os.environ.get("SM_CHANNEL_VALID"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_args()


def create_data_loaders(train_dir, valid_dir, test_dir, batch_size):
    """
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    """
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform)
    test_dataset = datasets.ImageFolder(test_dir, transform)

    dataset_sizes = {'train': len(train_dataset), 'valid': len(valid_dataset), 'test': len(test_dataset)}
    num_classes = len(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                                        
    data_loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}
                                        
    return data_loaders, dataset_sizes, num_classes



def train(model, data_loaders, criterion, optimizer, scheduler, dataset_sizes, device, num_epochs, hook):
    """
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    """
    since = time.time()
    model.to(device)
    phases = ["train", "valid"]
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == "train":
                model.train()  # Set model to training mode
                hook.set_mode(smd.modes.TRAIN)
            else:
                model.eval()   # Set model to evaluate mode
                hook.set_mode(smd.modes.EVAL)

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print(f"{phase}_loss: {epoch_loss:.4f}, {phase}_acc: {epoch_acc:.4f}")
            logger.info(f"{phase}_loss: {epoch_loss:.4f}, {phase}_acc: {epoch_acc:.4f}")

            # deep copy the model
            if phase == "valid" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()  # Create blank space between each epoch report

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best validation accuracy: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def test(model, test_loader, criterion, device, hook):
    """
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    """
    print("Testing model on the whole testing dataset")
    model.to(device)
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects / len(test_loader.dataset)
    # print(f"test_loss: {total_loss:4f}, test_accuracy: {total_acc:4f}")
    logger.info(f"test_loss: {total_loss:4f}, test_accuracy: {total_acc:4f}")


def net(num_classes, weights="IMAGENET1K_V2"):
    """
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    """
    model = models.resnet50(weights="IMAGENET1K_V2")

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model



def main(args):
    """
    TODO: Initialize a model by calling the net function
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_loaders, dataset_sizes, num_classes = create_data_loaders(args.train, args.valid, args.test, args.batch_size)
    model = net(num_classes)


    """
    TODO: Create your loss and optimizer
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(criterion)

    """
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    """
    model = train(model, data_loaders, criterion, optimizer, exp_lr_scheduler, dataset_sizes, device, args.epochs, hook)

    """
    TODO: Test the model to see its accuracy
    """
    test(model, data_loaders["test"], criterion, device, hook)

    """
    TODO: Save the trained model
    """
    t = time.time()
    time_stamp = time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(t))
    model_name = f"{model.__class__.__name__}-{time_stamp}.pth"
    torch.save(model, os.path.join(args.model_dir, model_name))


if __name__ == "__main__":
    """
    TODO: Specify all the hyperparameters you need to use to train your model.
    """
    args = create_args()
    logger.info(f"hyperparameters: {vars(args)}")
    main(args)
