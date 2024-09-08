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
    print("Artifact path:", artifact_path)

    artifact = wandb.Api().artifact(artifact_path, type="model")
    artifact.download(root=MODELS_DIR)

# Do the actual download: 
# download_artifact() # already done :-) 


def get_raw_model() -> ResNet:
    """This returns the model architecture without weights"""

    # overwrite final classifier layer with our own output layers
    N_CLASSES = 6

    model = resnet18(weights=None)  # not getting the weights from the original resnet18 here!
    model.fc = nn.Sequential(       # nb: make sure that this architecture is like in the Kaggle nb!
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES)
    )

    return model

def load_model() -> ResNet:
    """This returns the model with its wandb weights"""

    download_artifact()

    model = get_raw_model()
    model_state_dict_path = os.path.join(MODELS_DIR, MODEL_FILE_NAME)
    model_state_dict = torch.load(model_state_dict_path, map_location="cpu")    # loading weights _to_ the CPU.
    # The model_state_dict contains the _weights_ from the trained model!
    model.load_state_dict(model_state_dict, strict=True)    # nb: this was False earlier ...
    model.eval()

    return model


def load_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


# load_model()