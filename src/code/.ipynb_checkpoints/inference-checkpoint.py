import json
import logging
import sys
import os
import io
from PIL import Image

import torch
import torchvision
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def net(weights="IMAGENET1K_V2"):
    model = models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 133)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining model and loading weights to it.
def model_fn(model_dir): 
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(
            torch.load(f, map_location=device)
        )
    model.to(device).eval()
    return model


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "image/jpg"
    img = Image.open(io.BytesIO(request_body))
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    data = transformer(img)
    return data


# inference
def predict_fn(input_object, model):
    with torch.no_grad():
        # prediction = model(input_object)
        prediction = model(input_object.unsqueeze(0))
    return prediction


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)
