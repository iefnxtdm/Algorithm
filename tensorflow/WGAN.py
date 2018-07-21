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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x) #最后一层去掉sigmoid
        return x


gen = generator()
dis = discriminator()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
gen.to(device)
dis.to(device)
one = torch.FloatTensor([1]).cuda()
mone = one * -1
one.to(device)
mone.to(device)
#WGAN不用动量类optim, 减少梯度的漂移
optimizer_G = optim.RMSprop(gen.parameters(), lr=0.00005)
optimizer_D = optim.RMSprop(dis.parameters(), lr=0.00005)
# Configure data loader
os.makedirs('D:/mnist/', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('D:/mnist/', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])), batch_size=64, shuffle=True)
data_iter = iter(dataloader)
imp = next(data_iter)
print(imp[0].shape)

#print((img.shape())
gen_iterations = 0
print("finish load dataset")
for epoch in range(200):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        for p in dis.parameters():
            p.requires_grad = True
        # idx < 25 时 D 循环更新 25 次才会更新 G，用来保证 D 的网络大致满足 Wasserstein 距离
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = 5
        batch_size = 64
        noise = torch.Tensor(np.random.normal(0, 1, (batch_size, 100))).cuda()  # 随机生成noise
        """
            update D network
        """
        j = 0
        while j < Diters and i < len(dataloader):
            j += 1
            for p in dis.parameters():
                p.data.clamp_(-0.01, 0.01)  # 将判别器所有的参数截断到一个区间内
            img = data_iter.__next__()
            i += 1
            dis.zero_grad()
            # train real
            real_imgs = torch.Tensor(img[0]).cuda()
            real_loss = dis(real_imgs).mean(0)
            real_loss.backward(one)  # one mone用处是啥没弄清楚
            fake_loss = dis(gen(noise)).mean(0)
            fake_loss.backward(mone)
            d_loss = real_loss - fake_loss  # Wasserstein 距离
            optimizer_D.step()
            gen_iterations += 1
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f]" % (epoch, 200, i, len(dataloader),
                                                                             d_loss.item()))
            # update G network
        for p in dis.parameters():
            p.requires_grad = False
        gen.zero_grad()
        fake = gen(noise)
        g_loss = dis(fake).mean(0)
        g_loss.backward(one)
        optimizer_G.step()
        print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]" % (epoch, 200, i, len(dataloader), g_loss.item()))

        if gen_iterations % 400 <= 1:
            save_image(fake.data[:25].view(25, 1, 28, 28), 'D:/mnist/images-2/%d.png' % gen_iterations, nrow=5, normalize=True)