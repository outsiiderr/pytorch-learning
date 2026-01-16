import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#导包
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import math
import numpy as np
import torch.nn as nn


x_a = torch.rand(1000,1)*10.0
x_b=x_a-5.0
x_initial = torch.zeros(1, 1)#只需一个点
y_true=torch.ones_like(x_initial)
x_b.requires_grad_(True)
x_initial.requires_grad_(True)
#随机选取x并启用梯度

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(1,50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,50)
        self.fc4 = nn.Linear(50,1)#定义模型

    def forward(self, x):
        x=torch.tanh(self.fc1(x))
        x=torch.tanh(self.fc2(x))
        x=torch.tanh(self.fc3(x))
        x=self.fc4(x)
        return x#定义前向传播

def loss_ODE(x_b):#定义损失函数
    y_pred = model(x_b)
    dy_dx = torch.autograd.grad(outputs=y_pred,#启用自动求导，这个是用于操作的函数
                                inputs=x_b,#这个是用于操作的变量
                                grad_outputs=torch.ones_like(y_pred),#这里要设置为形状与outputs相同的张量，如果是标量，可以忽略
                                retain_graph=True,
                                create_graph=True)[0]
    wuli = dy_dx - 2 * x_b
    loss_ODE = torch.mean(wuli**2)
    return loss_ODE
def loss_IC(x_initial):#定义初始条件的损失函数
    y_initial_pred = model(x_initial)
    loss_IC= criterion(y_initial_pred,y_true)
    return loss_IC


model = MLP()#调用模型

optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion=nn.MSELoss()#定义优化器与损失函数

epochs=1000
total_train_losses=[]#用来记录损失值
for epoch in range(epochs):
    optimizer.zero_grad()  # 梯度清零，防止梯度累加
    loss_ode=loss_ODE(x_b)
    loss_ic=loss_IC(x_initial)
    total_loss=loss_ode+loss_ic
    total_train_losses.append(total_loss.item())#记录
    total_loss.backward()#反向传播
    optimizer.step()  # 更新参数

#绘制损失函数图像
plt.figure(figsize=(10, 5))
plt.plot(total_train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

x_test=torch.linspace(-5.0, 5.0, 1001).reshape(-1,1)#生成等间距的点
with torch.no_grad():
    y_test=model(x_test)
x_np=x_test.numpy().flatten()
y_np=y_test.numpy().flatten()

#绘制函数图像
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()




