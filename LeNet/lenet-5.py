import torch
from torchvision import datasets, transforms


class LeNet5(torch.nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.feature_extractor = torch.nn.Sequential(
            # Layer 1
            torch.nn.LazyConv2d(out_channels=6, kernel_size=5, stride=1, padding=0),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            # Layer 2
            torch.nn.LazyConv2d(out_channels=16, kernel_size=5, stride=1, padding=0),
            torch.nn.Sigmoid(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2),
            # Layer 3
            torch.nn.LazyConv2d(out_channels=120, kernel_size=5, stride=1, padding=0),
            torch.nn.Sigmoid(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features=84),
            torch.nn.Sigmoid(),  # Activation function -> shouldn't it be a RBF?
            torch.nn.LazyLinear(out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def fetch_mnist_data(batch_size):
    # Define a transform to normalize the data
    # Define a series of transformations to apply to the MNIST data
    transform = transforms.Compose(
        [
            # Resize all inputs.
            transforms.Resize((32, 32)),
            # Convert the PIL Image or numpy.ndarray to a PyTorch tensor
            transforms.ToTensor(),
            # Normalize the tensor with mean 0.5 and standard deviation 0.5
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Download and load the training data
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Create data loaders
    # Shuffling ensures that the data is presented in a different order each epoch, improving generalization
    # A batch size of 64 is a common choice as it balances memory usage and training speed
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


def main():
    print("LeNet-5")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 10

    model = LeNet5(num_classes).to(device)

    # Setting the loss function
    cost = torch.nn.CrossEntropyLoss()

    # Setting the optimizer with the model parameters and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Fetch data.
    train_loader, test_loader = fetch_mnist_data(batch_size=64)
    total_step = len(train_loader)

    # Train!
    # Iterate.
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 400 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

    # Evaluate!
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the 10000 test images: {accuracy:.2f} %")


if __name__ == "__main__":
    main()
