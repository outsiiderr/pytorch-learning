import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchvision
from torch import nn
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim import LBFGS
import matplotlib.pyplot as plt

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

content_image = Image.open(r'F:\data\content_style\街景.jpg')
style_image=Image.open(r'F:\data\content_style\星空.jpg')#导入图片

rgb_mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
rgb_std=torch.tensor([0.229, 0.224, 0.225]).to(device)

#定义预处理
def preprocess(img,image_shape):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(max(image_shape)),# 第一步：等比例缩放。
        torchvision.transforms.CenterCrop(image_shape),# 第二步：中心裁剪。
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transform(img).unsqueeze(0).to(device)#在最前面新增加批次维度

#定义后处理
def postprocess(img):
    img=img[0].to(rgb_std.device)#移除第一个维度
    img=torch.clamp(img.permute(1,2,0)*rgb_std+rgb_mean,0,1)#图像裁剪
               #重新排列维度           反标准化            裁剪的下限和上限
    return torchvision.transforms.ToPILImage()(img.permute(2,0,1))

# 提取特征
p_net=torchvision.models.vgg19(pretrained=True).to(device)#调用vgg19模型
p_net.eval()
style_layers,content_layers=[0,5,10,19,28],[25]#选取各自层数
net=nn.Sequential(*[p_net.features[i] for i in range\
    (max(content_layers+style_layers)+1)]).to(device)#确定需要用到的最后一层
for param in p_net.parameters():
    param.requires_grad = False#将用到的前29层提取出来

#两个图片分离
def extract_features(x,content_layers,style_layers):
    content_features=[]
    style_features=[]
    for i in range(len(net)):
        x=net[i](x)#将net中第i层的x张量传递给x
        if i in content_layers:
            content_features.append(x)
        if i in style_layers:
            style_features.append(x)#分离两个图片
    return content_features,style_features

def get_content(image_shape,device):
    content_x=preprocess(content_image,image_shape).to(device)
    content_y,_=extract_features(content_x,content_layers,style_layers)
    return content_x,content_y
def get_style(image_shape,device):
    style_x=preprocess(style_image,image_shape).to(device)
    _,style_y=extract_features(style_x,content_layers,style_layers)
    return style_x,style_y#实现了两个图片的分离

#定义损失函数
def content_loss(y_hat,y):
    loss=torch.square(y_hat-y.detach()).mean()#因为y是一个固定的数字，防止它进入计算图，用.detach将他固定
    return loss

def gram(x):
    channels,n=x.shape[1],x.numel()//x.shape[1]
    x=x.reshape((channels,n))
    gram=torch.matmul(x,x.T)/(channels*n)
    return gram

def style_loss(y_hat,gram_y):
    loss=torch.square(gram(y_hat)-gram_y.detach()).mean()
    return loss

def tv_loss(y_hat):#全变分去噪，去模糊
    return 0.5 * (torch.abs(y_hat[:,:,1:,:] - y_hat[:,:,:-1,:]).mean()+\
                 torch.abs(y_hat[:,:,:,1:] - y_hat[:,:,:,:-1]).mean())

# def tv_loss(y_hat):#全变分去噪，去模糊
#     diff_x=y_hat[:,:,1:,:]-y_hat[:,:,:-1,:]
#     diff_y=y_hat[:,:,:,1:                                                                                                                                                                                                                                                         ]-y_hat[:,:,:,:-1]
#     di_x=(diff_x.pow(2)).sum()#.pow()是逐元素操作，适用于任何形状的张量
#     di_y=(diff_y.pow(2)).sum()
#     tv_loss=torch.sqrt(di_x+di_y+1e-8)#1e-8是为了防止梯度是NaN
#     return tv_loss

weight_content,weight_style,weight_lv=1,1e4,10
# 定义总损失
def compute_loss(x,content_y_hat,content_y,style_y_hat,style_y_gram):
    content_l=[content_loss(y_hat,y)*weight_content for y_hat,y in zip(content_y_hat,content_y)]
    style_l=[style_loss(y_hat,y)*weight_style for y_hat,y in zip(style_y_hat,style_y_gram)]
    tv_l=tv_loss(x)*weight_lv
    l=sum(content_l+style_l+[tv_l])#将所有的损失全都写到一个列表之中，再将所有的元素求和
    return content_l,style_l,tv_l,l

#定义模型
class SynthesizedImage(nn.Module):
    def __init__(self,img_shape):
        super().__init__()
        self.weight=nn.Parameter(torch.randn(img_shape))#随机生成一个特定尺寸的张量，并将它作为参数

    def forward(self):
        return self.weight#我们只需要获得x图像的参数就好了

def get_inits(x,device,lr,style_y):
    gen_img=SynthesizedImage(x.shape).to(device)
    gen_img.weight.data.copy_(x.detach())#使用detach是使x脱离计算图，把x当前的像素值，复制到gen_img的可训练参数中，作为初始值
    # optimizer=torch.optim.Adam(gen_img.parameters(),lr=lr)#优化器
    optimizer=LBFGS(
        gen_img.parameters(),
        lr=lr,
        max_iter=20,#每次step最多迭代20次
        history_size=10,#存储10次历史梯度，省内存
        line_search_fn='strong_wolfe'#强wolfe条件线搜索
    )
    style_y_gram=[gram(y) for y in style_y]#因为风格图像的gram矩阵不再改变，因此提前求出来，防止后面重复计算
    return gen_img(),optimizer,style_y_gram

#训练模型
def train(x,content_y,style_y,lr,lr_decay_epoch,epochs):
    x,optimizer,style_y_gram=get_inits(x,device,lr,style_y)
    scheduler = StepLR(optimizer, lr_decay_epoch, gamma=0.8)#经过一定的迭代数后学习率乘0.8
    loss=[]
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            current_img = x#获取当前的图像张量
            content_y_hat, style_y_hat = extract_features(x, content_layers, style_layers)
            _,_,_,l = compute_loss(x, content_y_hat, content_y, style_y_hat, style_y_gram)
            l.backward()
            return l

        loss_l=optimizer.step(closure)
        loss.append(loss_l.item())
        scheduler.step()
        if(epoch%10==0):
            print('epoch=',epoch,'loss=',loss_l.item())
    return x,loss

#初始化合成图像
image_shape=(500,750)
content_x,content_y=get_content(image_shape,device)
_,style_y=get_style(image_shape,device)#不会用到style_x，因此忽略
output,loss=train(content_x,content_y,style_y,lr=1.0,lr_decay_epoch=10,epochs=30)
output_image=postprocess(output)

output_image.save(r"F:\data\content_style\结果\3.jpg")
plt.figure(figsize=(10,8))
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

plt.imshow(output_image)
plt.show()





