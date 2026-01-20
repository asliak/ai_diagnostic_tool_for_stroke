#!/usr/bin/env python3
import os, sys, json, base64
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torchvision import models, transforms
from io import BytesIO
from PIL import Image

CLASS_NAMES = [
    "Epidural",
    "Intraparenchymal",
    "Intraventricular",
    "Subarachnoid",
    "Subdural",
    "Any"
]

MODEL_PATH = "/home/nekodu-02/ee491/CerebrumScanner/model_path/hemorage_model.pth"

def encode_base64(img_uint8):
    buf = BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def load_model(device):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    return model

def preprocess(slice_2d):
    norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-6)
    img = (norm * 255).astype(np.uint8)
    rgb = np.stack([img]*3, axis=-1)
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485]*3, [0.229]*3)
    ])
    return tf(rgb), img

def main(nifti_path):
    device = torch.device("cpu")
    model = load_model(device)

    volume = nib.load(nifti_path).get_fdata()
    slice_results = []
    probs_all = []

    max_risk = -1
    max_idx = -1

    for i in range(volume.shape[2]):
        slice_2d = volume[:,:,i]
        if np.std(slice_2d) < 1e-3:
            continue

        inp, raw = preprocess(slice_2d)
        inp = inp.unsqueeze(0).to(device)

        with torch.no_grad():
            out = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

        out = (out / out.sum()) * 100
        probs = {c: round(float(p),2) for c,p in zip(CLASS_NAMES, out)}
        risk = max(out[:-1])

        if risk > max_risk:
            max_risk = risk
            max_idx = i

        probs_all.append(out)
        slice_results.append({
            "slice_index": i,
            "probabilities": probs,
            "image_base64": encode_base64(raw)
        })

    volume_probs = np.max(np.vstack(probs_all), axis=0)
    volume_probs = (volume_probs / volume_probs.sum()) * 100
    volume_probs = {c: round(float(p),2) for c,p in zip(CLASS_NAMES, volume_probs)}

    final = max({k:v for k,v in volume_probs.items() if k!="Any"}, key=volume_probs.get)

    print(json.dumps({
        "volume_level": {
            "probabilities": volume_probs,
            "final_diagnosis": final
        },
        "most_risky_slice_index": max_idx,
        "slice_level": slice_results
    }))

if __name__ == "__main__":
    main(sys.argv[1])
