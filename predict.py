
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from net import vgg16
test_pth=r'E:\xiangmu\VGGNet\train\cat\cat.1.jpg'#设置可以检测的图像
test=Image.open(test_pth)
'''处理图片'''
transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
image=transform(test)
'''加载网络'''
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")#CPU与GPU的选择
net =vgg16()#输入网络
model=torch.load(r"E:\xiangmu\VGGNet\road surface identification.13.pth",map_location=device)#已训练完成的结果权重输入
net.load_state_dict(model)#模型导入
net.eval()#设置为推测模式
image=torch.reshape(image,(1,3,224,224))#四维图形，RGB三个通
with torch.no_grad():
    out=net(image)
out=F.softmax(out,dim=1)#softmax 函数确定范围
out=out.data.cpu().numpy()
print(out)
a=int(out.argmax(1))#输出最大值位置
plt.figure()
list=['Cat','Dog']
plt.suptitle("Classes:{}:{:.1%}".format(list[a],out[0,a]))#输出最大概率的道路类型
plt.imshow(test)
plt.show()