import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import os

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(20), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

dataset_path = '/home/ctrigila/.cache/kagglehub/datasets/msambare/fer2013/versions/1'

train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transforms)
test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def create_tiny_vit_model(num_classes=7):
    model = models.vision_transformer.vit_b_16(pretrained=False) 
    
    # Modify the classifier to output 7 classes
    model.heads = model.heads[0]  
    model.heads = nn.Linear(model.heads.in_features, num_classes)  
    
    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False  

    # Unfreeze the final layer (classifier head)
    for param in model.heads.parameters():
        param.requires_grad = True  

    return model

def plot_and_save_metrics(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('Loss_Data_Augmentation.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy', color='green')
    plt.plot(test_accuracies, label='Testing Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('Accuracy_Data_Augmentation.png')
    plt.close()

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30):
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'Best_model_with_augmentation.pth')

    plot_and_save_metrics(train_losses, train_accuracies, test_accuracies)

    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_tiny_vit_model(num_classes=7).to(device)
    model.load_state_dict(torch.load('best_model.pth'))  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  

    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30)

    model.load_state_dict(torch.load('Best_model_with_augmentation.pth'))
    model.eval()

    # Evaluate the model on the test data
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = correct / total
    print(f'Test Accuracy : {test_accuracy:.4f}')

# Run the main function
if __name__ == "__main__":
    main()
