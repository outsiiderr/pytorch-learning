import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#调用GPU

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),#水平翻转
    transforms.RandomVerticalFlip(p=0.2),#垂直翻转
    transforms.RandomRotation(30),#旋转30度
    transforms.RandomResizedCrop(227),#随机裁剪并调整为227x227
    #原图输入224 × 224，实际上进行了随机裁剪，实际大小为227 × 227
    transforms.ToTensor(),#将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])#标准化
])#对训练集进行数据预处理及数据增强操作

test_transform = transforms.Compose([
    transforms.Resize(256),#将图像缩放至256*256
    transforms.CenterCrop(227),#将图像裁剪至227*227
    transforms.ToTensor(),#将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])#标准化
])#对测试集进行数据预处理及数据增强操作

train_dataset=datasets.ImageFolder(
    root=r'F:\data\rumen_2\flower\flower_copy\train_copy',
    transform=train_transform)#导入训练集
test_dataset=datasets.ImageFolder(
    root=r'F:\data\rumen_2\flower\flower_copy\test_copy',
    transform=test_transform)#导入测试集

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,#定义批量大小
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False)#定义数据加载器

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,96,11,4,0)
        self.conv2=nn.Conv2d(96,256,5,1,2)
        self.conv3=nn.Conv2d(256,384,3,1,1)
        self.conv4=nn.Conv2d(384,384,3,1,1)
        self.conv5=nn.Conv2d(384,256,3,1,1)
        #定义卷积层
        self.fc6=nn.Linear(6*6*256,4096)
        self.fc7=nn.Linear(4096,4096)
        self.fc8=nn.Linear(4096,5)
        #定义全连接层
        self.dropout=nn.Dropout(0.5)
        #定义dropout，随机删除一部分神经节点，防止过拟合

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x=F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=F.relu(self.conv5(x))
        x=F.max_pool2d(x,2)#13*13*256经过池化后变为6*6*256
        x=x.view(-1, 6*6*256)#展平
        x=F.relu(self.fc6(x))
        x=self.dropout(x)
        x=F.relu(self.fc7(x))
        x=self.dropout(x)
        x=self.fc8(x)
        return x
        #定义前向传播

model=CNN().to(device)#创建模型

criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)#优化

epochs=10
train_loss=[]#记录损失值
model.train()#训练模式
for epoch in range(epochs):
    total_loss=0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()#计算每个epoch的总损失
    average_loss=total_loss/len(train_loader)#每个epoch的平均损失
    train_loss.append(average_loss)#记录平均损失

model.eval()#评估模式
with torch.no_grad():#禁用梯度
    total=0
    correct=0
    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        _, preds = torch.max(outputs, 1)
        total += len(labels)
        correct += (labels==preds).sum().item()
    accuracy=100*correct/total#转化为百分数
    print("准确率是",accuracy)

plt.figure(figsize=(10, 5))
plt.plot(train_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()



