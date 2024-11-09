import cv2
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
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from PIL import Image
import matplotlib.pyplot as plt

def att(channal):
    # channal = np.uint8(channal)
    # print(channal)
    cv2.normalize(channal, channal, 0, 255, cv2.NORM_MINMAX)
    M = np.ones(channal.shape, np.uint8) * 255
    # print(channal.dtype,M.dtype)
    img_new = cv2.subtract(M, channal)
    return img_new


def process(img):
    # image = np.array(img)
    # print(image)
    image = np.uint8(img * 255)
    # image = image.transpose(1, 2, 0)
    # print(image.shape, image.dtype)
    # print(image)
    b, g, r = cv2.split(image)
    # print(b.shape)
    # print(b)
    b = att(b)
    g = att(g)
    r = att(r)
    new_image = cv2.merge([r, g, b])
    # print(new_image.shape)
    new_image = np.float32(new_image)
    new_image = new_image / 255.0
    return new_image


def lowlight(image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    AIP_net = model.AIP().cuda()
    AIP_net.load_state_dict(torch.load('code/snapshots/Epoch19.pth'))

    data_lowlight = Image.open(image_path)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    # start1 = time.time()
    a = process(data_lowlight)
    # print(a.shape)
    a = a.transpose(2, 0, 1)
    # # a = a[np.newaxis, ...]
    # # print(a.shape)
    b = torch.from_numpy(a)
    b = torch.unsqueeze(b, dim=0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.unsqueeze(0)
    mix = torch.cat((data_lowlight, b), dim=1)
    mix = mix.cuda()
    b = b.cuda()
    data_lowlight = data_lowlight.cuda()

    # data_lowlight = torch.from_numpy(data_lowlight).float()
    # data_lowlight = data_lowlight.permute(2,0,1)
    # data_lowlight = data_lowlight.cuda().unsqueeze(0)

    start = time.time()
    enhanced_image,_= AIP_net(data_lowlight, b, mix)

    # enhanced_image, r, y1, y2, y3, y4, y5, y6, y7 = DCE_net(data_lowlight, b, mix)

    # print(data_lowlight.dtype, data_lowlight.shape)
    # print(enhanced_image.dtype,enhanced_image.shape)
    end_time = (time.time() - start)#+end_time1
    print(end_time)
    image_path = image_path.replace('test_data', 'result')
    result_path = image_path
    if not os.path.exists(image_path.replace('/' + image_path.split("/")[-1], '')):
        os.makedirs(image_path.replace('/' + image_path.split("/")[-1], ''))
    torchvision.utils.save_image(enhanced_image, result_path)
    # torchvision.utils.save_image(y1, "y1.png")
    # torchvision.utils.save_image(y2, "y2.png")
    # torchvision.utils.save_image(y3, "y3.png")
    # torchvision.utils.save_image(y4, "y4.png")
    # torchvision.utils.save_image(y5, "y5.png")
    # torchvision.utils.save_image(y6, "y6.png")
    # torchvision.utils.save_image(y7, "y7.png")
    # torchvision.utils.save_image(r, "r.png")
    # torchvision.utils.save_image(rr, "rr.png")

if __name__ == '__main__':
    # test_images
    with torch.no_grad():
        filePath = 'code/data/test_data/'

        file_list = os.listdir(filePath)

        for file_name in file_list:
            test_list = glob.glob(filePath + file_name + "/*")
            for image in test_list:
                # image = image
                image = eval(repr(image).replace('\\', '/'))
                print(image)
                lowlight(image)

# python Zero-DCE_code/lowlight_test.py
