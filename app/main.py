from pydantic import BaseModel # Parent class for strict types in pydantic!

from fastapi import FastAPI, Depends, UploadFile, File 

# Python image library: for uploading images to fastabi
from PIL import image 


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
@app.post('/predict', response_model=Result)
async def predict(
    input_image: UploadFile = File(...), 
    model: ResNet = Depends(load_model)   # One whole Model for _each_ client! Expensive on memory!
): 