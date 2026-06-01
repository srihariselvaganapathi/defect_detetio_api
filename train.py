import torch
import torch.nn as nn
import torch.optim as optim
from model import DefectDetectorCNN
from dataset import train_loader

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device:{device}")

model=DefectDetectorCNN().to(device)

criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

epochs=5
print("starting the Training Engine...")

for epoch in range(epochs):
    running_loss=0.0

    for i,(images,labels) in enumerate(train_loader):
        images=images.to(device)
        labels=labels.float().view(-1,1).to(device)
        optimizer.zero_grad()
        predictions=model(images)
        loss=criterion(predictions,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

        if(i+1)%10==0:
            print(f"Epoch[{epoch+1}/{epochs}],Batch [{i+1}/{len(train_loader)}],loss:{loss.item():.4f}")
print("Training completed! The AI has learned to spot defects.")
torch.save(model.state_dict(),"defect_model.pth")
print("Model saved as defect_model.pth")