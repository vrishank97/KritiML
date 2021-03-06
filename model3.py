import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import math
import time
from torch.autograd import Variable
import torch.nn.functional as F

import torch
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchsummary import summary

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

BATCH_SIZE = 256
batch = 256
##train set loader
trainset = torchvision.datasets.ImageFolder(
    root='./train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)
classes = tuple(os.listdir('./train'))
number_to_name = {x: classes[x] for x in range(len(classes))}
name_to_number = {classes[x]: x for x in range(len(classes))}

#validation set loader
valid_scr = (np.genfromtxt('./val/val_annotations.txt', dtype='str'))[:, :2]
#y_ann = []
#for i in valid_scr[:, 1]:
#    y_ann.append(name_to_number[i])
y_ann = np.load("annotations_val.npy")
val_y = np.array(y_ann)
# val_x = np.zeros((1,3,64,64))
# i =0
# for fname in valid_scr[:, 0]:
#     temp = np.array([np.transpose(np.array(Image.open('./val/images/'+fname).convert('RGB')),(2,0,1))])
#     val_x = np.append(val_x,temp,axis=0)
#     i=i+1
#     if(i==500):
#         break
#     #print(fname)

val_x = np.array([np.array([np.transpose(np.array(Image.open(
    './val/images/'+fname).convert('RGB')), (2, 0, 1))]) for fname in valid_scr[:, 0]])

x = ((((val_x[:,0,:,:]))/255.0)-0.5)/0.5


tensor_data = torch.from_numpy(x)
tensor_data = tensor_data.float()
tensor_target = torch.from_numpy(val_y)
val_set = torch.utils.data.TensorDataset(tensor_data, tensor_target)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch,
                                         shuffle=True)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (5, 5), stride=1, padding=2)
        self.convadd = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnormadd = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*8*8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 200)
        self.bt1 = nn.BatchNorm1d(1024)
        self.bt2 = nn.BatchNorm1d(512)
        self.drp1 = nn.Dropout(p=0.5)
        self.drp2 = nn.Dropout2d(p=0.4)
        self.drp3 = nn.Dropout2d(p=0.4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        y = x
        x = self.convadd(x)
        x = self.batchnormadd(x)
        x = F.relu(x)
        x = x+y
        x = F.max_pool2d(x,(2,2))
        x = self.drp3(x)
        x = self.conv2(x)
        z = x
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = x+z
        x = F.max_pool2d(x,(2,2))
        x = self.drp2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2,2))
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # x = self.batchnorm1(x)
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # x = self.batchnorm2(x)
        # x  = self.drp1(x)
        # x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        # x = self.batchnorm3(x)
        x = x.view(-1, self.num_flat_features(x))
        x =self.drp1(x)
        x = F.relu(self.fc1(x))
        x = self.bt1(x)
        x = F.relu(self.fc2(x))
        x = self.bt2(x)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = LeNet().cuda()
print(summary(net, (3, 64, 64)))


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
EPOCHS = 20
t1 = time.time()
for epoch in range(EPOCHS):
    print("-- EPOCH:", epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        if i % 40 == 0:
            print("-- ITERATION:", i)
        input, target = data

        t0 = time.time()
        # wrap input + target into variables
        input_var = Variable(input).cuda()
        target_var = Variable(target).cuda()

        # compute output
        output = net(input_var)
        loss = criterion(output, target_var)

        # computer gradient + sgd step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t = time.time() - t0

        # print progress
        running_loss += loss.data[0].item()

        if i % 40 == 0:  # print every 2k mini-batches
            print("-- RUNNING_LOSS: ", running_loss / 50, "Time", t)
            running_loss = 0.0
    tikka = time.time() - t1
    print("time : ", tikka)
    cnt = 0
    aloo = 0
    for idx, data in enumerate(val_loader):
        net.eval()
        batch, label = data
        batch = batch.cuda()
        Y = net(batch)
        cnt += np.sum(np.equal(np.argmax(Y.cpu().detach().numpy(),
                                         axis=1), label.numpy()))
    #     # print("a")
    #     # print((np.argmax(Y.cpu().detach().numpy(), axis=1)))
    #     # print("b")
    #     # print(label.numpy())
    #     # print("c")
    #     # print((np.argmax(Y.cpu().detach().numpy(), axis=1).shape))
    #     # print("d")
    #     # print(label.numpy().shape)
    #     # print("e")
    #     # print((np.equal(np.argmax(Y.cpu().detach().numpy(),axis=1) ,label.numpy())))
    #     # print("f")
    #     # print((np.equal(np.argmax(Y.cpu().detach().numpy(), axis=1), label.numpy())).shape)
    #     # print("g")

    #     # print(cnt)

    print("vali bhaiiii!!!", epoch, "  ", cnt/(len(val_set)))
    if(aloo < cnt/(len(val_set))):
        torch.save(net.state_dict(), "baseline.pt")
    if(aloo < cnt/(len(val_set))):
        aloo = cnt/(len(val_set))

print('Finished Training')
