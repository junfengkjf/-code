import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

#准备数据集
train_data = torchvision.datasets.CIFAR10("./data/cifar-10-python",train=True,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./data/cifar-10-python",train=False,transform=torchvision.transforms.ToTensor())


train_dataloader=DataLoader(train_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)
#定义GPU.to(device)
#device=torch.device("cpu")
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f'训练数据集的长度为：{train_data_size}')
print(f'测试数据集的长度为：{test_data_size}')

#搭建神经网络
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1=nn.Sequential(
        nn.Conv2d(3,32,5,padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(32,32,5,padding=2),
        nn.MaxPool2d(2),
        nn.Conv2d(32,64,5,padding=2),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1024,64),
        nn.Linear(64,10)
        )

    def forward(self, x):
        x = self.module1(x)
        return x

tudui=Tudui()
#cuda可用时，使用
# if torch.cuda.is_available():
#     tudui=tudui.cuda()
#损失函数
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()

#优化器
learning_rate=1e-2
optimizer=torch.optim.SGD(tudui.parameters(),lr=learning_rate)

#设置训练网络的一些参数
total_trainstep=0
total_teststep=0
epoch=100

#添加tensorboard
writer=SummaryWriter("logs_CIFAR_train")
start_time=time.time()
for i in range(epoch):
    print(f"--------第{i+1}轮训练开始--------")#f字符串方法{}为表达式的值

    #训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs,targets=data
        outputs=tudui(imgs)#outputs得到的64*10的tensor，
        loss=loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_trainstep=total_trainstep+1

        if total_trainstep%100==0:
            end_time=time.time()
            print(end_time-start_time)
            print(f"训练次数：{total_trainstep}，loss:{loss}")
            #print(outputs)
            writer.add_scalar("train_loss",loss.item(),total_trainstep)


     #每一个epoch都测试
    tudui.eval()
    total_testloss=0
    total_accuracy=0


    with torch.no_grad():

        for data in test_dataloader:
            imgs,targets=data
            outputs=tudui(imgs)
            loss=loss_fn(outputs,targets)
            total_testloss=total_testloss+loss.item()
            #在第i个大的epoch里面，outputs的输出每一行，十列，每一列的元素最大的为分的类别，
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy


    print(f"整体数据集上的loss为{total_testloss}")
    print(f"整体数据集上的accuracy为{total_accuracy/test_data_size}")
    #可以绘制可视化由tensorboard
    writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_teststep)
    writer.add_scalar("test_loss",total_testloss,total_teststep)

    torch.save(tudui,f"tudui{i}.pth")
    #torch.save(tudui.state_dict(),f"tudui_{i}.pth")
    print("模型已经保存")

writer.close()
#-------注意---------
#调用GPU，需要修改网络模型，数据的输入，损失函数，需要使用.cuda
#利用训练好的模型，进行测试，相对路径要熟悉使用










