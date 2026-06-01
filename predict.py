import torch
from torchvision import transforms
from PIL import Image
from model import DefectDetectorCNN

# 1. Define the exact same math transformations used during training
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

def predict_defect(image_path):
    # 2. Hardware Check (Match what you trained on)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Load the Empty Brain and the Trained Weights
    model = DefectDetectorCNN().to(device)
    # Note: map_location handles loading a GPU model onto a CPU if needed
    model.load_state_dict(torch.load("defect_model.pth", map_location=device))
    
    # Put the model in "evaluation mode" (turns off training-specific features)
    model.eval() 

    # 4. Prepare the New Image
    image = Image.open(image_path)
    # Apply the resize/grayscale math and add a fake "batch" dimension
    image_tensor = image_transforms(image).unsqueeze(0).to(device) 

    # 5. Make the Prediction
    # We use torch.no_grad() because we don't need calculus/backprop just to guess
    with torch.no_grad():
        output = model(image_tensor)
        
        # The output is a number between 0 and 1 (from our Sigmoid layer)
        prediction_score = output.item() 
        
        # We trained it so that values closer to 1 are Defective
        if prediction_score > 0.5:
            print(f"Result: DEFECTIVE (Confidence: {prediction_score:.4f})")
        else:
            print(f"Result: OK (Confidence: {1 - prediction_score:.4f})")

# Let's test it! Point it to a specific image in your Kaggle test folder.
# Change this path to point to a real image on your computer
if __name__ == "__main__":
    test_image_path = r"C:\Users\Srihari\Desktop\defect_detetio_api\data\casting_data\casting_data\test\def_front\cast_def_0_118.jpeg"
    predict_defect(test_image_path)