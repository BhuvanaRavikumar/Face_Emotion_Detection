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

dataset_path = '/home/ctrigila/.cache/kagglehub/datasets/msambare/fer2013/versions/1'

train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transforms)
test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

def create_tiny_vit_model(num_classes=7, pretrained=True):
    model = models.vision_transformer.vit_b_16(pretrained=pretrained)  
    
    # Modify the classifier to output the correct number of classes
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    
    return model

def load_pretrained_model(weights_path, num_classes=7, device='cuda'):
    model = create_tiny_vit_model(num_classes=num_classes)

    state_dict = torch.load(weights_path, map_location=device)
    # Rename the keys to match the new layer names
    new_state_dict = {}
    for key, value in state_dict.items():
        if key == "heads.weight":  # Rename heads to heads.head
            new_state_dict["heads.head.weight"] = value
        elif key == "heads.bias":
            new_state_dict["heads.head.bias"] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict, strict=False)
    
    return model

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30):
    best_accuracy = 0.0
    train_losses, test_accuracies, train_accuracies = [], [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0
        
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
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Testing Accuracy: {test_accuracy:.4f}")
        
        # Save the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'Best_model_with_hyperparameter.pth')
    
    return train_losses, train_accuracies, test_accuracies

def plot_results(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Loss_Hyperparameter_Optimisation.png')
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Accuracy_Hyperparameter_Optimisation.png')
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_weights = 'Base_model.pth'  
    model = load_pretrained_model(pretrained_weights, num_classes=7, device=device).to(device)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)  

    train_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30
    )

    plot_results(train_losses, train_accuracies, test_accuracies)

    model.load_state_dict(torch.load('Best_model_with_hyperparameter.pth'))
    model.eval()
    
    # Evaluate the model on the test set
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

if __name__ == "__main__":
    main()
