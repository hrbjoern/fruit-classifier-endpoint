import wandb
import os
import torch 
from torch import nn 
from torchvision import transforms 
from torchvision.models import resnet18, ResNet 

from loadotenv import load_env

# TODO: to be removed later, when Dockerizing.
load_env(file_loc='/workspaces/fruit-classifier-endpoint/app/.env')


MODELS_DIR = 'models'
MODEL_FILE_NAME = 'model.pth'

CATEGORIES = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]


print(os.getenv('WANDB_API_KEY'))

def download_artifact(): 
    assert 'WANDB_API_KEY' in os.environ, \
        "please enter wandb api key as env var."

    wandb.login() # get access to artifacts registry

    #bjoern-opitz-none/banana_apple_orange/resnet18:v0
    wandb_org = os.getenv('WANDB_ORG')
    #print(wandb_org)
    wandb_project = os.environ.get('WANDB_PROJECT')
    wandb_model_name = os.environ.get('WANDB_MODEL_NAME')
    wandb_model_version = os.environ.get('WANDB_MODEL_VERSION')

    artifact_path = f'{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}'
    print(artifact_path)

    artifact = wandb.Api().artifact(artifact_path, type="model")
    artifact.download(root=MODELS_DIR)


download_artifact()


