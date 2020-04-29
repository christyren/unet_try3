import os
import datetime
import glob
import torch
import torch.utils.model_zoo as model_zoo
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Subset

from modules.metrics import runningScore
from modules.loss import CEDiceLoss, BCEDiceLoss
from modules.transforms import original_transform, teacher_transform
from models.unet import UNet
from modules.dice_loss import dice_coeff

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /unet

dataroot = os.path.join(os.path.dirname(base_path), "datasets")
if not os.path.exists(dataroot):
    os.mkdir(dataroot)

datasets = torchvision.datasets.VOCSegmentation(dataroot, year='2012', image_set='train', download=True,
                                                transform=original_transform, target_transform=teacher_transform)

train_size = len(datasets) * 9 // 10
val_size = len(datasets) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(datasets, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=21).to(device)
criterion = BCEDiceLoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# colabは相対パスがいいみたい
# logdir = "logs"
# logdir_path = os.path.join(base_path, logdir)
logdir_path = "./logs"
if not os.path.isdir(logdir_path):
    os.mkdir(logdir_path)
dt = datetime.datetime.now()
model_id = len(glob.glob(os.path.join(logdir_path, "{}{}{}*".format(dt.year, dt.month, dt.day))))
log_name = "{}{:02}{:02}_{:02}_{}".format(dt.year, dt.month, dt.day, model_id, model.__class__.__name__)
log_path = os.path.join(logdir_path, log_name)
writer = SummaryWriter(log_dir=log_path)

epochs = 20
n_classes = 21


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()

        optimizer.zero_grad()

        adjust_learning_rate(optimizer, epoch)

        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        writer.add_scalar("train_loss for Unet", loss.item(), (len(train_loader) * (epoch - 1) + batch_idx))  # 675*e+i

        if batch_idx % 20 == 0:
            # validation
            model.eval()
            with torch.no_grad():
                val_losses = 0.0
                val_dice_coeff = 0.0
                for idx, (data, target) in enumerate(val_loader):
                    data, target = data.cuda(), target.cuda()
                    embedded = model(data)
                    val_loss = criterion(embedded, target)
                    val_losses += val_loss
                    #
                    pred = torch.sigmoid(embedded)
                    pred = (pred > 0.5).float()
                    true_masks = target.to(device=device, dtype=torch.long)
                    val_dice_coeff += dice_coeff(pred, true_masks.float()).item()
            mean_dice_coeff = val_dice_coeff / len(val_loader)
            mean_val_loss = val_losses / len(val_loader)
            writer.add_scalar("validation/val_loss", mean_val_loss, (len(train_loader) * (epoch - 1) + batch_idx))
            writer.add_scalar("validation/dice_coeff", mean_dice_coeff, (len(train_loader) * (epoch - 1) + batch_idx))
            print('Train Epoch: {:>3} [{:>5}/{:>5} ({:>3.0f}%)]\ttrain loss: {:>2.4f}\tmean val loss: {:>2.4f}\tmean '
                  'dice coefficient: {:>2.4f}'.format(epoch, batch_idx * len(data),
                                                      len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                                                      loss.item(), mean_val_loss, mean_dice_coeff))


def save(epoch):
    checkpoint_path = os.path.join(base_path, "checkpoints")
    save_file = "checkpoint.pth.tar"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, log_name)):
        os.makedirs(os.path.join(checkpoint_path, log_name))
    save_path = os.path.join(checkpoint_path, log_name, save_file)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    model_load = False
    if model_load == True:
        start_epoch = 52
        epoch_range = range(start_epoch, epochs + 1)
    else:
        epoch_range = range(1, epochs + 1)
    dummy = torch.rand(4, 3, 256, 256).cuda()
    writer.add_graph(model, (dummy,))
    for epoch in epoch_range:
        train(epoch)
        save(epoch)
