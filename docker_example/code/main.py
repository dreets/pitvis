# global imports
import argparse
import albumentations as album
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torchvision import models
from tqdm import tqdm

# strongly typed
from albumentations import Compose
from numpy import ndarray
from pandas import DataFrame
from pathlib import Path
from PIL import Image
from torch import device
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Dict
from typing import List


def main(pt_input: Path, pt_output: Path):
    pt_current: Path = Path().absolute()
    pt_model: Path = pt_current.joinpath("model_weights.pth")
    tch_device = device("cuda")
    dl_loader: DataLoader = dataloader_initialise(pt_input=pt_input)
    md_model: Module = model_load(pt_model=pt_model, tch_device=tch_device)
    ls_preds: List[int] = ls_prediction_calculate(md_model=md_model, dl_loader=dl_loader, tch_device=tch_device)
    predictions_save(ls_preds=ls_preds, pt_output=pt_output)


def predictions_save(ls_preds: List[int], pt_output: Path):
    """save predictions as a csv"""
    ls_times: List[int] = [int_time for int_time in range(len(ls_preds))]
    dt_preds: Dict[str, List] = {"int_time": ls_times, "int_step": ls_preds}
    df_preds: DataFrame = pd.DataFrame(dt_preds)
    df_preds.to_csv(pt_output, mode="a", index=False)
    print("Prediction is saved.")


def ls_prediction_calculate(md_model: Module, dl_loader: DataLoader, tch_device: device) -> List[int]:
    """predict the classification of each frame of the video"""
    md_model.eval()
    ls_preds: List[int] = []
    tq_loop: tqdm = tqdm(dl_loader)
    for ts_imgs in tq_loop:
        ts_imgs: Tensor = ts_imgs.to(device=tch_device)
        with torch.set_grad_enabled(False):  # forward pass
            ts_cls_preds: Tensor = md_model(ts_imgs).to(device=tch_device)
            _, ts_cls_preds = torch.max(ts_cls_preds, 1)
        ls_preds.extend(ts_cls_preds.tolist())
    return ls_preds


def model_load(pt_model: Path, tch_device: device) -> Module:
    """load the model from the weights"""
    md_model: Module = models.resnet50(pretrained=True)
    md_model.fc = nn.Linear(md_model.fc.in_features, 15)
    md_model.load_state_dict(torch.load(str(pt_model)))
    md_model.to(device=tch_device)
    return md_model


def dataloader_initialise(pt_input: Path) -> DataLoader:
    """initialise inputs loader"""
    dataset: Dataset = PituitaryDataset(pt_input=pt_input)
    dl_loader: DataLoader = DataLoader(dataset=dataset)
    return dl_loader


class PituitaryDataset(Dataset):
    """create dataset for both steps and instruments"""
    def __init__(self, pt_input: Path):
        self.pt_input: Path = pt_input
        self.cp_transformation: Compose = album.Compose([album.Resize(height=224, width=224), ToTensorV2()])
        self.ls_pt_images: List[Path] = self.find_ls_pt_images()

    def __len__(self) -> int:
        """returns the total number of images for the chosen split"""
        return len(self.ls_pt_images)

    def __getitem__(self, int_index: int) -> Tensor:
        """return the tensor form of images and the corresponding class labels"""
        ts_img: Tensor = self.find_image_tensor(int_index=int_index)
        return ts_img

    def find_image_tensor(self, int_index: int) -> Tensor:
        """find the image tensor"""
        pt_img: Path = self.ls_pt_images[int_index]
        ay_img: ndarray = np.array(Image.open(pt_img).convert("RGB"), dtype=np.float32) / 255
        ts_img: Tensor = self.cp_transformation(image=ay_img)["image"]
        return ts_img

    def find_ls_pt_images(self) -> List[Path]:
        """find the list of image paths"""
        ls_pt_images: List[Path] = list(self.pt_input.iterdir())
        ls_pt_images.sort()
        return ls_pt_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pt_input",
        type=str,
        help="directory of the test dataset images './data/inputs/{int_video}/'"
    )
    parser.add_argument(
        "--pt_output",
        type=str,
        help="filepath of the annotation predictions './data/outputs/{int_video}.csv'"
    )
    args = vars(parser.parse_args())
    SystemExit(main(pt_input=Path(args["pt_input"]), pt_output=Path(args["pt_output"])))
