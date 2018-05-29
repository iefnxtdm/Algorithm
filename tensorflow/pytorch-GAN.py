# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc11 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.out = nn.Linear(1024, 784)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc11(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc3(x), 0.2, inplace=True)
        x = F.tanh(self.out(x))
        return x


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, input):
        x = input.view(input.shape[0], -1)
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        x = F.sigmoid(self.out(x))
        return x


gen = generator()
dis = discriminator()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
gen.to(device)
dis.to(device)
loss = nn.BCELoss()  # 二分类交叉熵
optimizer_G = optim.Adam(gen.parameters(), lr = 0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(dis.parameters(), lr = 0.0002, betas=(0.5, 0.999))

# Configure data loader
os.makedirs('D:/mnist/', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('D:/mnist/', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])), batch_size=64, shuffle=True)

print("finish load dataset")
for epoch in range(200):
    for i, (img, _) in enumerate(dataloader):
        batch = img.size(0)

        valid = torch.ones(batch, 1, dtype=torch.float, requires_grad=False)
        fake = torch.zeros(batch, 0, dtype=torch.float, requires_grad=False)
        real_imgs = torch.Tensor(img)  # (64,1,28,28)
        real_imgs = real_imgs.to(device)
        # train G
        optimizer_G.zero_grad()
        z = torch.Tensor(np.random.normal(0, 1, (batch, 100)))  # 随机生成
        gen_imgs = gen(z.cuda())
        g_loss = loss(dis(gen_imgs.cuda()), valid.cuda())  # 生成的用来欺骗分类器
        g_loss.backward(retain_graph=True)
        optimizer_G.step()

        # train D
        optimizer_D.zero_grad()
        real_loss = loss(dis(real_imgs.cuda()), valid.cuda())
        fake_loss = loss(dis(gen_imgs.cuda()), fake.cuda())
        d_loss = torch.add(real_loss, fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        if i % 300 == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 200, i, len(dataloader),
                                                                             d_loss.item(), g_loss.item()))
        batch_done = epoch * len(dataloader) + i
        if batch_done % 400 == 0:
            save_image(gen_imgs.data[:25].view(25, 1, 28, 28), 'D:/mnist/images/%d.png' % batch_done, nrow=5,
                       normalize=True)