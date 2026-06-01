import torch 
import torch.nn as nn
import torch.nn.functional as F

class DefectDetectorCNN(nn.Module):
    def __init__(self):
        super(DefectDetectorCNN,self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)

        self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)

        self.fcl1=nn.Linear(32*56*56,128)
        self.fcl2=nn.Linear(128,1)

    def forward(self,x):
        x=self.maxpool(F.relu(self.conv1(x)))
        x=self.maxpool(F.relu(self.conv2(x)))
        x=x.view(-1,32*56*56)
        x=F.relu(self.fcl1(x))
        x=torch.sigmoid(self.fcl2(x))
        return x

if __name__=="__main__":
    model=DefectDetectorCNN()
    print(model)    

    