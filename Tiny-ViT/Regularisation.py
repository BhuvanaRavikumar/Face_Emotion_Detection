import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
def load_fer2013(dataset_path):
    train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transforms)
    test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=test_transforms)
    return train_dataset, test_dataset

class TinyViTWithDropout(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.3, saved_weights_path='Base_model.pth'):
        super(TinyViTWithDropout, self).__init__()
        # Initialize Tiny ViT model without pre-trained weights
        self.model = models.vision_transformer.vit_b_16(pretrained=False)

        # Modify the classification head
        in_features = self.model.heads.head.in_features  # Get the number of input features
        self.model.heads = nn.Sequential(
            nn.Dropout(p=dropout_rate),  
            nn.Linear(in_features, num_classes) 
        )

        # Load previously saved weights
        state_dict = torch.load(saved_weights_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("heads."):
                new_key = key.replace("heads.", "heads.head.")  # Adjust the naming
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30):
    best_accuracy = 0.0
    train_losses, train_accuracies, test_accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Compute training metrics
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'Best_model_with_regularization.pth')

    plot_training_results(train_losses, train_accuracies, test_accuracies)
    return model

def plot_training_results(train_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('Loss_Regularisation.png')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='green')
    plt.plot(epochs, test_accuracies, label='Testing Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.savefig('Accuracy_Regularisation.png')
    
    plt.tight_layout()
    plt.show()

def main():
    dataset_path = '/home/ctrigila/.cache/kagglehub/datasets/msambare/fer2013/versions/1'
    train_dataset, test_dataset = load_fer2013(dataset_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model and load the saved weights
    model = TinyViTWithDropout(num_classes=7, dropout_rate=0.3, saved_weights_path='Base_model.pth').to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)  # Include L2 regularization

    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30)

    # Evaluate the best model
    model.load_state_dict(torch.load('Best_model_with_regularization.pth'))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Test Accuracy : {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
