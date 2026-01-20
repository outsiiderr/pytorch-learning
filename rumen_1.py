import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df=pd.read_csv('F:/data/rumen_1/data2.csv')#导入数据集
df= df.fillna(df.mean())#填补缺失值

x=df.iloc[:,1:].values
y=df.iloc[:,0].values

def maxmin(x):
    x_min=x.min(axis=0)
    x_max=x.max(axis=0)
    x_scaled=(x-x_min)/(x_max-x_min)
    return x_scaled#定义归一化

x=maxmin(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.3,
                                               random_state=42,
                                               shuffle=True)#划分数据集
print(pd.DataFrame(x_train))
print(pd.DataFrame(x_test))
x_train_tensor,x_test_tensor=torch.from_numpy(x_train).float(),torch.from_numpy(x_test).float()
y_train_tensor,y_test_tensor=torch.from_numpy(y_train).float(),torch.from_numpy(y_test).float()
#将numpy数组转变为tensor
#注意：.float()才是正确的，得到的是张量，.float()返回的是方法对象


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(7,100)
        self.fc2=nn.Linear(100,50)
        self.fc3=nn.Linear(50,50)

    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))
        return x                             #定义前馈神经网络

model=SimpleNN()#使用模型

criterion=nn.BCELoss()#定义损失函数：用于二分类
optimizer = optim.Adam(model.parameters(), lr=0.0005)#随即下降优化器，此处有学习率

epochs=5000
train_losses=[]#用来记录损失值
for epoch in range(epochs):
    optimizer.zero_grad()  # 梯度清零，防止梯度累加
    y_pred_tensor=model(x_train_tensor)#前向传播
    loss=criterion(y_pred_tensor.squeeze(),y_train_tensor)
    #目标张量是一维，预测张量是二维，利用squeeze()将他转换成一维
    train_losses.append(loss.item())  # 记录损失值
    loss.backward()#反向传播
    optimizer.step()#更新
    if epoch%1000==0:
        print(f'Epoch {epoch} Loss {loss.item():.4f}')



model.eval()#设置为评估模式
with torch.no_grad():#禁用梯度
    y_pred_all=model(x_test_tensor)
    y_pred_classes = (y_pred_all > 0.5).float().squeeze()
    accuracy=(y_pred_classes==y_test_tensor).float().mean()#计算准确率
    print("准确率是",accuracy)

plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
