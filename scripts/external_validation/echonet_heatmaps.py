"""Save EchoNet-Measurements heatmaps (sigmoid outputs) for every input DICOM, to read the model's response at a given
beat. Preprocessing and model loading copied from the official scripts (inference_Doppler_image.py and
inference_MV_EperA.py): same ECG masking, same crop (region y0 for single-point models, row 342 for E/A).
Run in the 'echonet' env with cwd = the Measurement folder."""
import sys
from pathlib import Path

import numpy as np
import pydicom
import torch
from pydicom.pixel_data_handlers.util import convert_color_space
from torchvision.models.segmentation import deeplabv3_resnet50

sys.path.insert(0, ".")
from utils import get_coordinates_from_dicom  # noqa: E402

IN = Path(r"C:\Claude\Doppler\validation\echonet_in")
OUT = Path(r"C:\Claude\Doppler\validation\echonet_out\heatmaps")
OUT.mkdir(parents=True, exist_ok=True)
JOBS = [("tdi", "latevel", 1), ("tdi", "medevel", 1), ("tr", "trvmax", 1), ("mitral", "mvpeak_2c", 2)]


def load_image(f):
    ds = pydicom.dcmread(f)
    im = ds.pixel_array
    pi = ds.PhotometricInterpretation
    if pi == "MONOCHROME2":
        im = np.stack((im,) * 3, axis=-1)
    elif pi == "YBR_FULL_422" and im.ndim == 3:
        im = convert_color_space(arr=im, current="YBR_FULL_422", desired="RGB")
        im[np.logical_and(im[:, :, 1] > 200, im[:, :, 0] < 100), :] = 0
    elif pi == "RGB":
        im = im.copy()
        im[np.logical_and(im[:, :, 1] > 200, im[:, :, 0] < 100), :] = 0
    return im, ds


for task, model, ncls in JOBS:
    net = deeplabv3_resnet50(num_classes=ncls)
    w = torch.load(f"./weights/Doppler_models/{model}_weights.ckpt", map_location="cpu")
    print(model, net.load_state_dict({k.replace("m.", ""): v for k, v in w.items()}))
    net.eval()
    for f in sorted((IN / task).glob("*.dcm")):
        im, ds = load_image(f)
        region = get_coordinates_from_dicom(ds)[0]
        y0 = int(region[(0x0018, 0x601A)].value)
        crop0 = 342 if ncls == 2 else y0
        t = torch.tensor(im[crop0:]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            h = torch.sigmoid(net(t)["out"]).squeeze(0).numpy().astype(np.float16)  # (C, H, W)
        np.savez_compressed(OUT / f"{f.stem}__{model}.npz", heat=h, crop0=crop0)
        print("  ", f.stem, h.shape)
