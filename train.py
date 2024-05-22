import argparse
import torch
import torch.nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
from common import TRAIN_FOLDER, VAL_FOLDER
from balancedaccuracy import BalancedAccuracy
from network import Net
from transforms import TrainingTransform, ValidationTransform

# NOTE: You do not need to change this file
# Make sure your other code works around this

BATCH_SIZE = 8


def train(args):
    # Setup the ImageFolder Dataset
    trainset = torchvision.datasets.ImageFolder(
        TRAIN_FOLDER, transform=TrainingTransform
    )
    validationset = torchvision.datasets.ImageFolder(
        VAL_FOLDER, transform=ValidationTransform
    )

    # Extract number of classes to define network architecture
    nClasses = len(trainset.classes)

    # Create data loader
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    validationloader = DataLoader(
        validationset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Create the network, the optimizer and the loss function
    net = Net(nClasses)
    optim = torch.optim.Adam(net.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    # Train some epochs
    bacc = BalancedAccuracy(nClasses)
    for epoch in range(int(args.epochs)):
        for loader in [trainloader, validationloader]:
            if loader == trainloader:
                net.train()
                training = True
                label = "train:"
            else:
                net.eval()
                training = False
                label = "val:  "

            total_loss = 0
            total_cnt = 0
            bacc.reset()

            bar = tqdm(loader)
            for batch, labels in bar:
                optim.zero_grad()

                out = net(batch)
                assert out.shape[0] == BATCH_SIZE
                assert out.shape[1] == nClasses

                bacc.update(out, labels)

                loss = criterion(out, labels)
                total_loss += loss.item()
                total_cnt += batch.shape[0]

                loss.backward()
                optim.step()

                bar.set_description(
                    f"{label}   {epoch+1:3}/{int(args.epochs)}   loss={100.0 * total_loss / total_cnt:10.5f}    bacc={100.0 * bacc.getBACC():.2f}%"
                )

        # Save a checkpoint after each epoch
        torch.save({"model": net.state_dict(), "classes": trainset.classes}, "model.pt")


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)

args = parser.parse_args()

train(args)
