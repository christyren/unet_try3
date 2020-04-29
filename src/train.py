import os
import datetime
import glob
from PIL import Image
import torch
import torch.utils.model_zoo as model_zoo
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Subset

from modules.loss import CEDiceLoss, BCEDiceLoss
from modules.transforms import original_transform, teacher_transform
from models.unet import UNet
from models.fcn import fcn8s
from models.segnet import segnet
from modules.dice_loss import dice_coeff
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # /unet


dataroot = os.path.join(os.path.dirname(base_path), "datasets")
if not os.path.exists(dataroot):
    os.mkdir(dataroot)

datasets = torchvision.datasets.VOCSegmentation(dataroot, year='2012', image_set='train', download=True, transform=original_transform, target_transform=teacher_transform)


train_size = len(datasets)*9//10
val_size = len(datasets) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(datasets, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=21).to(device)
criterion = BCEDiceLoss().to(device)
model_fcn8 = fcn8s().to(device)
criterion_fcn8 = BCEDiceLoss().to(device)
model_segnet = segnet().to(device)
criterion_segnet = BCEDiceLoss().to(device)


optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer_fcn8 = optim.Adam(model_fcn8.parameters(), lr=0.001)
optimizer_segnet = optim.Adam(model_segnet.parameters(), lr=0.001)

# colabは相対パスがいいみたい
# logdir = "logs"
# logdir_path = os.path.join(base_path, logdir)
logdir_path = "./logs"
if not os.path.isdir(logdir_path):
    os.mkdir(logdir_path)
dt = datetime.datetime.now()

# log writer for unet
model_id = len(glob.glob(os.path.join(logdir_path, "{}{}{}*".format(dt.year, dt.month, dt.day))))
log_name = "{}{:02}{:02}_{:02}_{}".format(dt.year, dt.month, dt.day, model_id, model.__class__.__name__)
log_path = os.path.join(logdir_path, log_name)
writer_unet = SummaryWriter(log_dir=log_path)

# log writer for fcn8
log_name = "{}{:02}{:02}_{:02}_{}".format(dt.year, dt.month, dt.day, model_id, model_fcn8.__class__.__name__)
log_path = os.path.join(logdir_path, log_name)
writer_fcn8 = SummaryWriter(log_dir=log_path)

# log writer for segnet
log_name = "{}{:02}{:02}_{:02}_{}".format(dt.year, dt.month, dt.day, model_id, model_segnet.__class__.__name__)
log_path = os.path.join(logdir_path, log_name)
writer_segnet = SummaryWriter(log_dir=log_path)

epochs = 2


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        model_fcn8.train()
        model_segnet.train()

        optimizer.zero_grad()
        optimizer_fcn8.zero_grad()
        optimizer_segnet.zero_grad()

        adjust_learning_rate(optimizer, epoch)
        adjust_learning_rate(optimizer_fcn8, epoch)
        adjust_learning_rate(optimizer_segnet, epoch)

        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        output_fcn8 = model_fcn8(data)
        loss_fcn8 = criterion_fcn8(output_fcn8, target)
        loss_fcn8.backward()
        optimizer_fcn8.step()

        output_segnet = model_segnet(data)
        loss_segnet = criterion_segnet(output_segnet, target)
        loss_segnet.backward()
        optimizer_segnet.step()

        writer_unet.add_scalar("train_loss for Unet", loss.item(), (len(train_loader)*(epoch-1)+batch_idx)) # 675*e+i
        writer_fcn8.add_scalar("train_loss for fcn8", loss_fcn8.item(), (len(train_loader)*(epoch-1)+batch_idx)) # 675*e+i
        writer_segnet.add_scalar("train_loss for segnet", loss_segnet.item(), (len(train_loader)*(epoch-1)+batch_idx)) # 675*e+i

        if batch_idx % 20 == 0:
            # validation
            model.eval()
            model_fcn8.eval()
            model_segnet.eval()
            with torch.no_grad():
                val_losses, val_losses_fcn8, val_losses_segnet = 0.0, 0.0, 0.0
                val_dice_coeff, val_dice_coeff_fcn8, val_dice_coeff_segnet = 0.0, 0.0, 0.0
                for idx, (data, target) in enumerate(val_loader):
                    data, target = data.cuda(), target.cuda()
                    # unet
                    embedded = model(data)
                    val_loss = criterion(embedded, target)
                    val_losses += val_loss
                    pred = torch.sigmoid(embedded)
                    pred = (pred > 0.5).float()
                    true_masks = target.to(device=device, dtype=torch.long)
                    val_dice_coeff += dice_coeff(pred, true_masks.float()).item()
                    # fcn8
                    embedded_fcn8 = model_fcn8(data)
                    val_loss_fcn8 = criterion(embedded_fcn8, target)
                    val_losses_fcn8 += val_loss_fcn8
                    pred = torch.sigmoid(embedded_fcn8)
                    pred = (pred > 0.5).float()
                    val_dice_coeff_fcn8 += dice_coeff(pred, true_masks.float()).item()
                    # segnet
                    embedded_segnet = model_segnet(data)
                    val_loss_segnet = criterion(embedded_segnet, target)
                    val_losses_segnet += val_loss_segnet
                    pred = torch.sigmoid(embedded_segnet)
                    pred = (pred > 0.5).float()
                    val_dice_coeff_segnet += dice_coeff(pred, true_masks.float()).item()
            mean_dice_coeff = val_dice_coeff / len(val_loader)
            mean_val_loss = val_losses / len(val_loader)
            mean_dice_coeff_fcn8 = val_dice_coeff_fcn8 / len(val_loader)
            mean_val_loss_fcn8 = val_losses_fcn8 / len(val_loader)
            mean_dice_coeff_segnet= val_dice_coeff_segnet / len(val_loader)
            mean_val_loss_segnet = val_losses_segnet / len(val_loader)
            writer_unet.add_scalar("validation/val_loss_unet", mean_val_loss, (len(train_loader) * (epoch - 1) + batch_idx))
            writer_unet.add_scalar("validation/dice_coeff_unet", mean_dice_coeff, (len(train_loader) * (epoch - 1) + batch_idx))
            writer_fcn8.add_scalar("validation/val_loss_fcn8", mean_val_loss_fcn8, (len(train_loader) * (epoch - 1) + batch_idx))
            writer_fcn8.add_scalar("validation/dice_coeff_fcn8", mean_dice_coeff_fcn8, (len(train_loader) * (epoch - 1) + batch_idx))
            writer_segnet.add_scalar("validation/val_loss_segnet", mean_val_loss_segnet, (len(train_loader) * (epoch - 1) + batch_idx))
            writer_segnet.add_scalar("validation/dice_coeff_segnet", mean_dice_coeff_segnet, (len(train_loader) * (epoch - 1) + batch_idx))
            print('Train Epoch for UNet: {:>3} [{:>5}/{:>5} ({:>3.0f}%)]\ttrain_loss for Unet: {:>2.4f}\t    train_loss for '
                  'fcn8: {:>2.4f}\t    train_loss for segnet: {:>2.4f}'.format(
                epoch,
                batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), loss_fcn8.item(), loss_segnet.item()))


def save(epoch):
    checkpoint_path = os.path.join(base_path, "checkpoints")
    save_file = "checkpoint_unet.pth.tar"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, log_name)):
        os.makedirs(os.path.join(checkpoint_path, log_name))
    save_path = os.path.join(checkpoint_path, log_name, save_file)
    torch.save(model.state_dict(), save_path)
    save_file = "checkpoint_fcn8.pth.tar"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, log_name)):
        os.makedirs(os.path.join(checkpoint_path, log_name))
    save_path = os.path.join(checkpoint_path, log_name, save_file)
    torch.save(model_fcn8.state_dict(), save_path)
    save_file = "checkpoint_segnet.pth.tar"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, log_name)):
        os.makedirs(os.path.join(checkpoint_path, log_name))
    save_path = os.path.join(checkpoint_path, log_name, save_file)
    torch.save(model_segnet.state_dict(), save_path)


if __name__ == "__main__":
    model_load = False
    if model_load == True:
        start_epoch = 52
        epoch_range = range(start_epoch, epochs+1)
    else:
        epoch_range = range(1, epochs+1)
    dummy = torch.rand(4, 3, 256, 256).cuda()
    writer_unet.add_graph(model, (dummy,))
    writer_fcn8.add_graph(model_fcn8, (dummy, ))
    writer_segnet.add_graph(model_segnet, (dummy, ))
    print('# Model_UNet parameters:', sum(param.numel() for param in model.parameters()))
    print('# Model_fcn8 parameters:', sum(param.numel() for param in model_fcn8.parameters()))
    print('# Model_segnet parameters:', sum(param.numel() for param in model_segnet.parameters()))
    for epoch in epoch_range:
        train(epoch)
        save(epoch)
