from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
from model import DefectDetectorCNN

# 1. Initialize the Web Server
app = FastAPI(title="Industrial Defect Detection API")

# 2. Load the Brain (Do this once when the server starts)
device = torch.device("cpu") # Forcing CPU for the web server to keep it stable
model = DefectDetectorCNN().to(device)
model.load_state_dict(torch.load("defect_model.pth", map_location=device))
model.eval()

# 3. The Math Transformations
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# 4. The API Endpoint
@app.post("/predict")
async def predict_image(file: UploadFile):
    try:
        # Read the uploaded image file from memory
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Apply math and add batch dimension
        image_tensor = image_transforms(image).unsqueeze(0).to(device)
        
        # Make Prediction
        with torch.no_grad():
            output = model(image_tensor)
            score = output.item()
            
            # Format the output exactly how a frontend developer wants it
            if score > 0.5:
                result = "OK"
                confidence = float(score)
            else:
                result = "DEFECTIVE"
                confidence = float(1 - score)
                
        # Return a clean JSON response
        return JSONResponse(content={
            "filename": file.filename,
            "prediction": result,
            "confidence_score": round(confidence, 4)
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)