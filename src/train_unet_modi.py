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
from models.unet_modi import UNet_half
from models.unet_modi import UNet_2, UNet_3
from modules.dice_loss import dice_coeff

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /unet

dataroot = os.path.join(os.path.dirname(base_path), "datasets")
if not os.path.exists(dataroot):
    os.mkdir(dataroot)

datasets = torchvision.datasets.VOCSegmentation(dataroot, year='2012', image_set='train', download=True,
                                                transform=original_transform, target_transform=teacher_transform)

# train_loader = torch.utils.data.DataLoader(datasets, batch_size=4, shuffle=True)
train_size = len(datasets) * 9 // 10
val_size = len(datasets) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(datasets, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet_half(n_channels=3, n_classes=21).to(device)
criterion = BCEDiceLoss().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

model_small = UNet_2(n_channels=3, n_classes=21).to(device)
criterion_small = BCEDiceLoss().to(device)
optimizer_small = optim.Adam(model_small.parameters(), lr=0.001)

model_simple = UNet_3(n_channels=3, n_classes=21).to(device)
criterion_simple = BCEDiceLoss().to(device)
optimizer_simple = optim.Adam(model_simple.parameters(), lr=0.001)
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
writer_half = SummaryWriter(log_dir=log_path)

log_name = "{}{:02}{:02}_{:02}_{}".format(dt.year, dt.month, dt.day, model_id, model_small.__class__.__name__)
log_path = os.path.join(logdir_path, log_name)
writer_small = SummaryWriter(log_dir=log_path)

log_name = "{}{:02}{:02}_{:02}_{}".format(dt.year, dt.month, dt.day, model_id, model_simple.__class__.__name__)
log_path = os.path.join(logdir_path, log_name)
writer_simple = SummaryWriter(log_dir=log_path)

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
        model_small.train()
        model_simple.train()
        optimizer.zero_grad()
        optimizer_small.zero_grad()
        optimizer_simple.zero_grad()

        adjust_learning_rate(optimizer, epoch)
        adjust_learning_rate(optimizer_small, epoch)
        adjust_learning_rate(optimizer_simple, epoch)

        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        output_small = model_small(data)
        loss_small = criterion(output_small, target)
        loss_small.backward()
        optimizer_small.step()

        output_simple = model_simple(data)
        loss_simple = criterion(output_simple, target)
        loss_simple.backward()
        optimizer_simple.step()

        writer_half.add_scalar("train_loss for Unet_cut_half", loss.item(),
                               (len(train_loader) * (epoch - 1) + batch_idx))  # 675*e+i
        writer_small.add_scalar("train_loss for Unet simplify input-output layers", loss_small.item(),
                                (len(train_loader) * (epoch - 1) + batch_idx))  # 675*e+i
        writer_simple.add_scalar("train_loss for Unet simplify number of channels", loss_simple.item(),
                                 (len(train_loader) * (epoch - 1) + batch_idx))  # 675*e+i
        if batch_idx % 20 == 0:
            # validation
            model.eval()
            model_small.eval()
            model_simple.eval()
            with torch.no_grad():
                val_losses, val_losses_small, val_losses_simple = 0.0, 0.0, 0.0
                val_dice_coeff, val_dice_coeff_small, val_dice_coeff_simple = 0.0, 0.0, 0.0
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

                    embedded_small = model_small(data)
                    val_loss_small = criterion(embedded_small, target)
                    val_losses_small += val_loss_small
                    #
                    pred = torch.sigmoid(embedded_small)
                    pred = (pred > 0.5).float()
                    val_dice_coeff_small += dice_coeff(pred, true_masks.float()).item()

                    embedded_simple = model_simple(data)
                    val_loss_simple = criterion(embedded_simple, target)
                    val_losses_simple += val_loss_simple
                    #
                    pred = torch.sigmoid(embedded_simple)
                    pred = (pred > 0.5).float()
                    val_dice_coeff_simple += dice_coeff(pred, true_masks.float()).item()
            mean_dice_coeff = val_dice_coeff / len(val_loader)
            mean_val_loss = val_losses / len(val_loader)
            mean_dice_coeff_small = val_dice_coeff_small / len(val_loader)
            mean_val_loss_small = val_losses_small / len(val_loader)
            mean_dice_coeff_simple = val_dice_coeff_simple / len(val_loader)
            mean_val_loss_simple = val_losses_simple / len(val_loader)
            writer_half.add_scalar("validation/val_loss", mean_val_loss, (len(train_loader) * (epoch - 1) + batch_idx))
            writer_half.add_scalar("validation/dice_coeff", mean_dice_coeff,
                                   (len(train_loader) * (epoch - 1) + batch_idx))

            writer_small.add_scalar("validation/val_loss for simplify in-out layer", mean_val_loss_small,
                                    (len(train_loader) * (epoch - 1) + batch_idx))
            writer_small.add_scalar("validation/dice_coeff for simplify in-out layer", mean_dice_coeff_small,
                                    (len(train_loader) * (epoch - 1) + batch_idx))

            writer_simple.add_scalar("validation/val_loss for simplify number of channels", mean_val_loss_simple,
                                     (len(train_loader) * (epoch - 1) + batch_idx))
            writer_simple.add_scalar("validation/dice_coeff for simplify number of channels", mean_dice_coeff_simple,
                                     (len(train_loader) * (epoch - 1) + batch_idx))
            print('Train Epoch: {:>3} [{:>5}/{:>5} ({:>3.0f}%)]\ttrain loss: {:>2.4f}\tmean val loss: {:>2.4f}\tmean '
                  'dice coefficient: {:>2.4f}'.format(epoch, batch_idx * len(data),
                                                      len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                                                      loss.item(), mean_val_loss, mean_dice_coeff))
            print('Train Epoch  for simplify input-output layers: {:>3} [{:>5}/{:>5} ({:>3.0f}%)]\ttrain loss: {'
                  ':>2.4f}\tmean val loss: {:>2.4f}\tmean dice coefficient: {:>2.4f}'.format(epoch, batch_idx * len(
                data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                                                                                             loss_small.item(),
                                                                                             mean_val_loss_small,
                                                                                             mean_dice_coeff_small))
            print('Train Epoch  for simplify number of channels: {:>3} [{:>5}/{:>5} ({:>3.0f}%)]\ttrain loss: {'
                  ':>2.4f}\tmean val loss: {:>2.4f}\tmean dice coefficient: {:>2.4f}'.format(epoch, batch_idx * len(
                data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                                                                                             loss_simple.item(),
                                                                                             mean_val_loss_simple,
                                                                                             mean_dice_coeff_simple))


def save(epoch):
    checkpoint_path = os.path.join(base_path, "checkpoints")
    save_file = "checkpoint.pth.tar"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, log_name)):
        os.makedirs(os.path.join(checkpoint_path, log_name))
    save_path = os.path.join(checkpoint_path, log_name, save_file)
    torch.save(model.state_dict(), save_path)
    save_file = "checkpoint_small.pth.tar"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, log_name)):
        os.makedirs(os.path.join(checkpoint_path, log_name))
    save_path = os.path.join(checkpoint_path, log_name, save_file)
    torch.save(model_small.state_dict(), save_path)
    save_file = "checkpoint_simple.pth.tar"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, log_name)):
        os.makedirs(os.path.join(checkpoint_path, log_name))
    save_path = os.path.join(checkpoint_path, log_name, save_file)
    torch.save(model_simple.state_dict(), save_path)


if __name__ == "__main__":
    model_load = False
    if model_load == True:
        start_epoch = 52
        epoch_range = range(start_epoch, epochs + 1)
    else:
        epoch_range = range(1, epochs + 1)
    dummy = torch.rand(4, 3, 256, 256).cuda()
    writer_half.add_graph(model, (dummy,))
    writer_small.add_graph(model_small, (dummy,))
    writer_simple.add_graph(model_simple, (dummy,))
    print('# Model_half parameters:', sum(param.numel() for param in model.parameters()))
    print('# Model_small parameters:', sum(param.numel() for param in model_small.parameters()))
    print('# Model_simple parameters:', sum(param.numel() for param in model_simple.parameters()))

    for epoch in epoch_range:
        train(epoch)
        save(epoch)
