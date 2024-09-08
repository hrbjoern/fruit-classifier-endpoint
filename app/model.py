import wandb
import os
import torch 
from torch import nn 
from torchvision import transforms 
from torchvision.models import resnet18, ResNet 

from loadotenv import load_env

load_env(file_loc='/workspaces/fruit-classifier-endpoint/app/.env')

MODELS = 'models'
MODEL_FILE_NAME = 'model.pth'

CATEGORIES = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]


#print(os.getenv('WANDB_API_KEY'))

#def download_artifact(): 