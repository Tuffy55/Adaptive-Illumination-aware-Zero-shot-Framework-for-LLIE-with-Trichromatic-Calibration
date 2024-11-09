import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms
import cv2


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def att(channal):
    cv2.normalize(channal, channal, 0, 255, cv2.NORM_MINMAX)
    M = np.ones(channal.shape, np.uint8) * 255
    img_new = cv2.subtract(M, channal)
    return img_new


def process(img):
    image = np.array(img)
    image = np.uint8(image * 255)
    image = image.transpose(1, 2, 0)
    b, g, r = cv2.split(image)
    b = att(b)
    g = att(g)
    r = att(r)
    new_image = cv2.merge([r, g, b])
    new_image = np.float32(new_image)
    new_image = new_image / 255.0
    return new_image


def rgb2h(img):
    hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)
    eps = 1e-8
    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + eps))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue = hue / 6
    hue = hue.unsqueeze(1)
    return hue


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    AIP_net = model.AIP().cuda()


    if config.load_pretrain == True:
        AIP_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    L_rgb = Myloss.L_rgb()
    L_nc = Myloss.L_nc()

    L_br = Myloss.L_br(16, 0.65)

    L_es = Myloss.L_es()

    optimizer = torch.optim.Adam(AIP_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    AIP_net.train()

    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            attention = []
            p = 0
            for n in img_lowlight:


                a = process(n)

                a = a.transpose(2, 0, 1)
                b = torch.from_numpy(a)

                if p == 0:
                    attention = torch.unsqueeze(b, dim=0)
                    m = torch.cat((n, b), dim=0)
                    mix = torch.unsqueeze(m, dim=0)
                    p = p + 1

                elif p > 0:
                    a = torch.unsqueeze(b, dim=0)
                    attention = torch.cat((attention, a), dim=0)

                    m = torch.cat((n, b), dim=0)
                    m = torch.unsqueeze(m, dim=0)
                    mix = torch.cat((mix, m), dim=0)
                    p = p + 1

            mix = mix.cuda()
            attention = attention.cuda()
            img_lowlight = img_lowlight.cuda()

            enhanced_image, A = AIP_net(img_lowlight, attention, mix)


            Loss_es = L_es(A)

            loss_nc = torch.mean(L_nc(enhanced_image, img_lowlight))

            loss_rgb = torch.mean(L_rgb(enhanced_image))

            loss_br = torch.mean(L_br(enhanced_image))



            loss = 800*Loss_es + 50*loss_nc + 5*loss_rgb + 10*loss_br
            #

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(AIP_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save(AIP_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="code/data/train_data/")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="code/snapshots/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default="code/snapshots/Epoch99.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)
