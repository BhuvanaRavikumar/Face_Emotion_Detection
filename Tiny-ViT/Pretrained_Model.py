import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import os

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

# Load the FER2013 dataset from directories
dataset_path = '/home/ctrigila/.cache/kagglehub/datasets/msambare/fer2013/versions/1'

train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transforms)
test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Tiny ViT model
def create_tiny_vit_model(num_classes=7):
    model = models.vision_transformer.vit_b_16(pretrained=True)  
    
    # Modify the classifier to output 7 classes 
    model.heads = model.heads[0]  # This gets the last Linear layer
    model.heads = nn.Linear(model.heads.in_features, num_classes)  
    
    # Freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False  

    # Unfreeze the final layer 
    for param in model.heads.parameters():
        param.requires_grad = True  
    return model

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30):
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
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
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'Base_model.pth')
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_tiny_vit_model(num_classes=7).to(device)  # Initialize the model
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  
    
    model = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30)
    
    model.load_state_dict(torch.load('Base_model.pth'))
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
    print(f'Test Accuracy: {test_accuracy:.4f}')

if __name__ == "__main__":
    main()
