import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset.dataset import Dataset
from models.model import Model

import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()

def train(args, model, dataloader, loss_fn, optimizer, device):
    model.train()
    
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

    return model


def valid(args, model, dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            acc = (outputs.argmax(1) == labels).float().mean()

    return loss, acc


def main():
    args = arg_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = Dataset(root=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.epochs):

        model = train(args, model, dataloader, loss_fn, optimizer, device)

        loss, acc = valid(args, model, dataloader, loss_fn, device)

        torch.save(model.state_dict(), f'./model_{epoch}.pth')


if __name__ == '__main__':
    main()

    

    







