import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(784, 512),
            nn.ReLU(),

            # Second hidden layer 
            nn.Linear(512, 256),
            nn.ReLU(),

            # Output layer
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

def train_model(model, train_loader, test_loader, learning_rate, epochs):
    # Loss function: Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Fixed typo: grade -> grad
            outputs = model(images)
            loss = criterion(outputs, labels)  # Fixed spacing
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}] ',
                      f'Loss: {running_loss/100:.4f}',
                      f'Accuracy: {100 * correct / total:.2f}%')
                running_loss = 0
                
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

def main():
    # Data loading and transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model and training parameters
    model = MNISTClassifier()
    learning_rate = 0.01
    epochs = 10

    # Train the model
    train_model(model, train_loader, test_loader, learning_rate, epochs)

if __name__ == '__main__':
    main()
