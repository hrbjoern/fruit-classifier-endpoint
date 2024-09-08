from pydantic import BaseModel # Parent class for strict types in pydantic!
from fastapi import FastAPI, Depends, UploadFile, File 
# Python image library: for uploading images to fastabi
from PIL import Image 
import io 

import torch
import torch.nn.functional as F

from torchvision import transforms 
from torchvision.models import resnet18, ResNet 

from app.model import load_model, load_transforms, CATEGORIES

# Strictly typing: 
# category ~ label, 
# confidence ~ probability. 
class Result(BaseModel):
    category: str
    confidence: float 

# Create an instance for the endpoint: 
app = FastAPI()


# Asynchronous: Non-blocking server-client interaction! 
# e.g. _not_ possible in Flask!
@app.post('/predict', response_model=Result) # Remember: Result is a subclass of the pydantic BaseModel. For POST!
async def predict(
        input_image: UploadFile = File(...), 
        # Make sure the trained model is loaded: 
        # The output of load_model() is assigned to model (..!), which is of type ResNet.
        model: ResNet = Depends(load_model),   # One whole Model for _each_ client! Expensive on memory!
        transforms: transforms.Compose = Depends(load_transforms)
    ) -> Result: 
    
    # Read the uploaded image: 
    image = Image.open(io.BytesIO(await input_image.read()))

    # Convert RGBA to RGB:
    if image.mode == 'RGBA':
        image.convert('RGB')
    
    # apply the transformations to the image
    # We use unsqueeze(0) to define a batch size of 1 to feed the tensor to the model.
    image = transforms(image).unsqueeze(0)

    # Make the prediction: 
    with torch.no_grad(): # not saving the 
        outputs = model(image)
        # TODO: set up a breakpoint (pdb?) to understand outputs[0] and dim=0 (?)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    # Predicted label:
    category = CATEGORIES[predicted_class.item()]

    return Result(category=category, confidence=confidence.item())