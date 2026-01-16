import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#定义内部点
n_pde=8000
x_pde=torch.rand(n_pde,1)*2.0
y_pde=torch.rand(n_pde,1)*2.0
t_pde=torch.rand(n_pde,1)
x_pde=x_pde.requires_grad_(True).to(device)
y_pde=y_pde.requires_grad_(True).to(device)
t_pde=t_pde.requires_grad_(True).to(device)

#定义初始点
n_ic=2000
x_ic=torch.rand(n_ic,1)*2.0
y_ic=torch.rand(n_ic,1)*2.0
t_ic=torch.zeros(n_ic).view(-1,1)
x_ic=x_ic.to(device)
y_ic=y_ic.to(device)
t_ic=t_ic.to(device)

#定义边界点
n_bc=500
point_left=torch.cat([torch.zeros(n_bc,1).to(device),
                      torch.rand(n_bc,1).to(device)*2.0,
                      torch.rand(n_bc,1).to(device)],
                     dim=1)
point_right=torch.cat([torch.full((n_bc,1),2.0).to(device),
                       torch.rand(n_bc,1).to(device)*2.0,
                       torch.rand(n_bc,1).to(device)],
                      dim=1)
point_bottom=torch.cat([torch.rand(n_bc,1).to(device)*2.0,
                        torch.zeros(n_bc,1).to(device),
                        torch.rand(n_bc,1).to(device)],
                       dim=1)
point_top=torch.cat([torch.rand(n_bc,1).to(device)*2.0,
                     torch.full((n_bc,1),2.0).to(device),
                     torch.rand(n_bc,1).to(device)],
                    dim=1)

#定义模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 1)


    def forward(self,x):
        x=torch.tanh(self.fc1(x))
        x=torch.tanh(self.fc2(x))
        x=torch.tanh(self.fc3(x))
        x=torch.tanh(self.fc4(x))
        x=self.fc5(x)
        return x

model=MLP().to(device)
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
scheduler = StepLR(optimizer, step_size=8000, gamma=0.5)

# scheduler_1=torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor=0.5,
#     patience=200,
#     threshold=1e-3,
#     threshold_mode='rel',
#     cooldown=0,
#     min_lr=1e-8,
#     eps=1e-8
# )
# scheduler_2=torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode='min',
#     factor=0.8,
#     patience=200,
#     threshold=0.1,
#     threshold_mode='',
#     cooldown=0,
#     min_lr=1e-8,
#     eps=1e-8
# )


#定义损失函数
def loss_pde(x,y,t):
    u_pde=model(torch.cat([x,y,t],dim=1))
    du_dx=torch.autograd.grad(
        u_pde,x,
        torch.ones_like(u_pde),
        retain_graph=True,
        create_graph=True
    )[0]
    du_dx_2=torch.autograd.grad(
        du_dx,x,
        torch.ones_like(du_dx),
        retain_graph=True,
        create_graph=True
    )[0]
    du_dy = torch.autograd.grad(
        u_pde, y,
        torch.ones_like(u_pde),
        retain_graph=True,
        create_graph=True
    )[0]
    du_dy_2 = torch.autograd.grad(
        du_dy, y,
        torch.ones_like(du_dy),
        retain_graph=True,
        create_graph=True
    )[0]
    du_dt=torch.autograd.grad(
        u_pde,t,
        torch.ones_like(u_pde),
        retain_graph=True,
        create_graph=True
    )[0]
    f_pde=(988.5*torch.exp(-t)*torch.sin(torch.pi * x / 2) * torch.sin(torch.pi * y *10)\
           -torch.exp(-2*t)*(torch.sin(torch.pi*x/2)**2)*(torch.sin(torch.pi*y*10)**2))
    l_pde=du_dt-du_dx_2-du_dy_2-u_pde**2-f_pde
    loss=torch.mean(l_pde**2)
    return loss

def loss_ic(x, y, t):
    u_ic = model(torch.cat([x, y, t], dim=1))
    l_ic = u_ic - torch.sin(torch.pi * x / 2) * torch.sin(torch.pi * y * 10)
    loss = torch.mean(l_ic ** 2)
    return loss

def loss_bc():
    u_bc_left = model(point_left)
    u_bc_right = model(point_right)
    u_bc_bottom = model(point_bottom)
    u_bc_top = model(point_top)
    loss=torch.mean(u_bc_left**2)+torch.mean(u_bc_right**2)\
         +torch.mean(u_bc_bottom**2)+torch.mean(u_bc_top**2)
    return loss

weights=[1.0,100.0,50.0]


epochs=40000
total_loss=[]
loss_p_w=[]
loss_i_w=[]
loss_b_w=[]
for epoch in range(epochs):
    optimizer.zero_grad()
    loss_p=loss_pde(x_pde,y_pde,t_pde)
    loss_i=loss_ic(x_ic,y_ic,t_ic)
    loss_b=loss_bc()
    train_loss=weights[0]*loss_p+weights[1]*loss_i+weights[2]*loss_b
    total_loss.append(train_loss.item())#记录损失值 b
    loss_p_w.append(loss_p.item())
    loss_i_w.append(loss_i.item())
    loss_b_w.append(loss_b.item())
    current_lr = optimizer.param_groups[0]['lr']


    if epoch % 100 == 0:
        print('epoch=', epoch, 'loss=',train_loss.item(),'loss_pde=', loss_p.item(), 'loss_ic=', loss_i.item(),\
              'loss_bc=', loss_b.item(), 'lr=', current_lr)
        # if epoch % 1000 == 0:
        #     grad_norm = []
        #     for L in [loss_p, loss_i, loss_b]:
        #         # with torch.no_grad():
        #             grads = torch.autograd.grad(L, model.parameters(), retain_graph=True)
        #             total_norm = 0.0
        #             for g in grads:
        #                 if g is not None:
        #                     para_norm = g.norm(2)
        #                     total_norm += para_norm ** 2.0
        #             total_norm = total_norm ** 0.5
        #             grad_norm.append(total_norm.item())
        #
        #     # 避免除以0
        #     eps = 1e-8
        #     grad_norm = [max(gn, eps) for gn in grad_norm]
        #     norm_mean = sum(grad_norm) / len(grad_norm)
        #     weights = np.array(norm_mean) / np.array([loss_p.item(), loss_i.item(), loss_b.item()])



    train_loss.backward()
    optimizer.step()
    scheduler.step()
    # if epoch<10000:
    #     scheduler_1.step(train_loss.item())
    # else:
    #     scheduler_2.step(train_loss.item())


df=pd.read_csv(r'F:\data\PINN\submission.csv')
x_test=torch.tensor(df['x'].values,dtype=torch.float32).view(-1,1)
y_test=torch.tensor(df['y'].values,dtype=torch.float32).view(-1,1)
t_test=torch.tensor(df['t'].values,dtype=torch.float32).view(-1,1)
x_test=x_test.to(device)
y_test=y_test.to(device)
t_test=t_test.to(device)
u_pred=model(torch.cat([x_test,y_test,t_test],dim=1))
print('u_pred=',u_pred)

plt.figure(figsize=(10,8))
plt.plot(total_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,8))
plt.plot(loss_p_w)
plt.xlabel('Epoch')
plt.ylabel('Loss_pde')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,8))
plt.plot(loss_i_w)
plt.xlabel('Epoch')
plt.ylabel('Loss_ic')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,8))
plt.plot(loss_b_w)
plt.xlabel('Epoch')
plt.ylabel('Loss_bc')
plt.grid(True)
plt.show()

