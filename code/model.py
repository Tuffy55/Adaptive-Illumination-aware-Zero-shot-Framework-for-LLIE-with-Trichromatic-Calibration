import numpy as np
import torch.nn as nn
import torch
import torchvision

def default_conv(in_channels, out_channels, kernel_size, bias=True):
	return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class PALayer(nn.Module):
	def __init__(self, channel):
		super(PALayer, self).__init__()
		self.pa = nn.Sequential(
			nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.pa(x)
		return x * y,y


class CALayer(nn.Module):
	def __init__(self, channel):
		super(CALayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.ca = nn.Sequential(
			nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.ca(y)

		return x * y


class Block(nn.Module):
	def __init__(self, conv, dim, kernel_size, ):
		super(Block, self).__init__()
		self.conv1 = conv(dim, dim, kernel_size, bias=True)
		self.act1 = nn.ReLU(inplace=True)
		self.conv2 = conv(dim, dim, kernel_size, bias=True)
		#
		self.calayer = CALayer(dim)
		self.palayer = PALayer(dim)

	def forward(self, x):
		res = self.act1(self.conv1(x))
		res = res + x
		res = self.conv2(res)
		res = self.calayer(res)
		res,a = self.palayer(res)

		res += x
		return res


class Group(nn.Module):
	def __init__(self, conv, dim, kernel_size, blocks):
		super(Group, self).__init__()
		modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
		modules.append(conv(dim, dim, kernel_size))
		self.gp = nn.Sequential(*modules)

	def forward(self, x):
		res = self.gp(x)
		res += x
		return res


class AIP(nn.Module):
	def __init__(self, gps=2, blocks=1, conv=default_conv):
		super(AIP, self).__init__()
		self.gps = gps
		self.dim = 64
		kernel_size = 3
		pre_process = [conv(6, self.dim, kernel_size)]
		assert self.gps == 2
		self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
		self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
		#
		self.ca = nn.Sequential(*[
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
			nn.ReLU(inplace=True),
			nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
			nn.Sigmoid()
		])
		self.palayer = PALayer(self.dim)

		post_precess = [
			conv(self.dim, self.dim, kernel_size),
			conv(self.dim, 3, kernel_size)]

		self.pre = nn.Sequential(*pre_process)
		self.post = nn.Sequential(*post_precess)


	def forward(self, x1, att, mix):
		x = self.pre(mix)
		res1 = self.g1(x)
		res2 = self.g2(res1)
		#
		w = self.ca(torch.cat([res1, res2], dim=1))
		w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
		out = w[:, 0, ::] * res1 + w[:, 1, ::] * res2
		out,a1 = self.palayer(out)
		r = self.post(out)

		r = r * att

		en = x1 + r * (torch.pow(x1, 2) - x1)
		en = en + r*(torch.pow(en,2)-en)
		en = en + r*(torch.pow(en,2)-en)
		enhance_image_1 = en + r*(torch.pow(en,2)-en)
		en = enhance_image_1 + r*(torch.pow(enhance_image_1,2)-enhance_image_1)
		en = en + r*(torch.pow(en,2)-en)
		en = en + r*(torch.pow(en,2)-en)
		enhance_image = en + r*(torch.pow(en,2)-en)

		return enhance_image, r



if __name__ == "__main__":
	net = AIP(gps=2, blocks=1)
	print(net)