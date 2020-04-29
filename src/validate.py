import os

import torchvision
import yaml
import torch
import argparse
import timeit
import numpy as np

from torch.utils import data
from models.unet import UNet

# from ptsemseg.models import get_model
# from ptsemseg.loader import get_loader
from modules.metrics import runningScore
from modules.utils import convert_state_dict
from modules.transforms import original_transform, teacher_transform

torch.backends.cudnn.benchmark = True


def validate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    # data_loader = get_loader(cfg["data"]["dataset"])
    model_id = "20200404_00_UNet"

    checkpoint_path = "../checkpoints/{}/checkpoint.pth.tar".format(model_id)

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataroot = os.path.join(os.path.dirname(base_path), "datasets")
    if not os.path.exists(dataroot):
        os.mkdir(dataroot)
    n_classes = 21

    model = UNet(n_channels=3, n_classes=21).to(device)  # .cuda()

    datasets = torchvision.datasets.VOCSegmentation(dataroot, year='2012', image_set='train', download=False,
                                                    transform=original_transform, target_transform=teacher_transform)

    # valloader = data.DataLoader(loader, batch_size=cfg["training"]["batch_size"], num_workers=8)
    valloader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False)
    running_metrics = runningScore(n_classes)


    # Setup Model

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    # model.to(device)

    flag = False
    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()

        images = images.to(device)

        if flag:
            outputs = model(images)

            # Flip images in numpy (not support in tensor)
            outputs = outputs.data.cpu().numpy()
            flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
            flipped_images = torch.from_numpy(flipped_images).float().to(device)
            outputs_flipped = model(flipped_images)
            outputs_flipped = outputs_flipped.data.cpu().numpy()
            outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0

            pred = np.argmax(outputs, axis=1)
        else:
            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()

        gt = labels.numpy()

        if True:
            elapsed_time = timeit.default_timer() - start_time
            print(
                "Inference time \
                  (iter {0:5d}): {1:3.5f} fps".format(
                    i + 1, pred.shape[0] / elapsed_time
                )
            )
        running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    validate()
