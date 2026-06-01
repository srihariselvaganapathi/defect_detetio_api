import torch
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

#Must resize the image exactly same(224x224)
#convert physial image into PyTorch Tensors.

image_transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),#Casting defects are easier to spot in black & white
    transforms.ToTensor()
])

print("Transform locked in.Ready to process pixel matrices.")

from torch.utils.data import DataLoader

train_dataset=datasets.ImageFolder(root=r'C:\Users\Srihari\Desktop\defect_detetio_api\data\casting_data\casting_data\train',transform=image_transforms)
test_dataset=datasets.ImageFolder(root=r'C:\Users\Srihari\Desktop\defect_detetio_api\data\casting_data\casting_data\test',transform=image_transforms)

train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

print(f"Loaded {len(train_dataset)} training image and {len(test_dataset)} testing image.")

images,labels=next(iter(train_loader))

print(f"Batch shape:{images.shape}")
plt.imshow(images[0].squeeze(),cmap='gray')
plt.title(f"Label:{'Defective' if labels[0]==0 else 'ok'}")
plt.show()
