import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import random_split, DataLoader

#하이퍼 파라미터
BATCH_SIZE = 32
EPOCHS = 5  
PATIENCE = 3

def get_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    return model
#시간관계상 early stopping 적용
def train(model, trainloader, valloader, optimizer, criterion, device, name):
    print(f"\nStarting training for {name}...")
    
    os.makedirs('models', exist_ok=True)
    save_path = f"models/{name}.pth"
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}], Train Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(valloader)
        print(f"--> Epoch [{epoch + 1}/{EPOCHS}] Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
                
    print(f"Finished training. Best model saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    model_a = get_model().to(device)
    optimizer_a = optim.AdamW(model_a.parameters(), lr=1e-4, weight_decay=0.01)
    train(model_a, trainloader, valloader, optimizer_a, criterion, device, "resnet50_A_AdamW")

    model_b = get_model().to(device)
    optimizer_b = optim.RMSprop(model_b.parameters(), lr=1e-4, momentum=0.9)
    train(model_b, trainloader, valloader, optimizer_b, criterion, device, "resnet50_B_RMSprop")

if __name__ == "__main__":
    main()