import torch.nn as nn
import torch
torch.cuda.empty_cache()
import numpy as np
import cv2
#Model is as per the DCNN proposed in the paper 

class DeConv(nn.Module):
    def __init__(self):
        super(DeConv, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,49), stride=1, padding=(0,24))#Win-k+2P+1=Wout
        #self.drop=nn.Dropout2d(0.4)
        torch.nn.init.xavier_uniform(self.cnn1.weight)
        self.T = nn.Tanh()
        
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(49,1), stride=1, padding=(24,0))
        torch.nn.init.xavier_uniform(self.cnn2.weight)
        self.T2 = nn.Tanh()
        #self.drop2=nn.Dropout2d(0.4)
        
        # Linear O/p
        self.cnn3=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,padding=0,bias=False)
        torch.nn.init.xavier_uniform(self.cnn3.weight)
    def forward(self, x):
        # Convolution 1
        
        
        out = self.cnn1(x)
        
        out = self.T(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        
        out = self.T2(out)
        out = self.cnn3(out)
        
        return out
class ODeConv(nn.Module):
    def __init__(self):
        super(ODeConv, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1,49), stride=1, padding=(0,24))#Win-k+2P+1=Wout
        #self.drop=nn.Dropout2d(0.4)
        torch.nn.init.xavier_uniform(self.cnn1.weight)
        self.T = nn.ReLU()#Tanh
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(49,1), stride=1, padding=(24,0))
        #self.drop2=nn.Dropout2d(0.4)
        torch.nn.init.xavier_uniform(self.cnn2.weight)
        self.T2 = nn.ReLU()
        
        self.cnn3=nn.Conv2d(in_channels=40,out_channels=40,kernel_size=1,padding=0)
        torch.nn.init.xavier_uniform(self.cnn3.weight)
        self.cnn4=nn.Conv2d(in_channels=40,out_channels=30,kernel_size=17,padding=8)
        torch.nn.init.xavier_uniform(self.cnn4.weight)
        self.T4 = nn.ReLU()
        self.cnn5=nn.Conv2d(in_channels=30,out_channels=1,kernel_size=9,padding=4)
        torch.nn.init.xavier_uniform(self.cnn5.weight)
    def forward(self, x):
        # Convolution 1
        
        
        out = self.cnn1(x)
        #out=self.T(out)
        out = self.T(out)
        # Convolution 2 
        out = self.cnn2(out)
        #out=self.T2(out)
        out = self.T2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        out = self.T4(out)
        out = self.cnn5(out)
        return out

class DeoisingCNN(nn.Module):
    def __init__(self, num_channels, num_of_layers=17):
        super(DeoisingCNN, self).__init__()#Inheriting properties of nn.Module class
        l=[]
        #padding 1 as kernel is of size 3 and we need the o/p of CNN block to give same size img and no maxpooling or BN in this layer
        first_layer=nn.Sequential(nn.Conv2d(in_channels=1, out_channels=num_channels, kernel_size=3, padding=1, bias=False),nn.ReLU(inplace=True))
        l.append(first_layer)
        #All blocks in b/w the first and last are the same having same i.e having depth and no maxpooling layer
        for _ in range(num_of_layers-2):
            second_layer = nn.Sequential(
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1))#0.2
            l.append(second_layer)
        #Final layer is similar to the first CNN block
        l.append(nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, padding=1, bias=False))
        self.mdl = nn.ModuleList(l)
    def forward(self, x):
        out = self.mdl[0](x)
        for i in range(len(self.mdl) - 2):
            out = self.mdl[i + 1](out)
        out = self.mdl[-1](out)
        return out

def psnr(i1, i2):
    mse = torch.mean( (i1 - i2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

import torchvision.datasets as dset
import torchvision.transforms as transforms
train_image_folder = dset.ImageFolder(root='train',transform=transforms.ToTensor())
from torch.utils.data import DataLoader
from torch.autograd import Variable
model=DeConv().cuda()
criterion = nn.MSELoss()
from torch import optim
import matplotlib.pyplot as plt
optimizer = optim.Adam(model.parameters(), lr=0.001)#lr=0.001
#Clean images will be corrupted and model will learn the distribution of noise given the corrupted images
loader_train = DataLoader(dataset=train_image_folder,batch_size=10, shuffle=True)
epochs=50#1 imgae is for 5 epochs
iter=0
best_loss=np.inf
l=[]
itr=[]
#model1=torch.load('noise3.pkl')
from skimage.measure import compare_ssim as ssim

for epoch in range(epochs):
    for i, (im1,l1) in enumerate(loader_train):
        x=im1.data.numpy()
        
        blur=[]
        for k in range(im1.size()[0]):
            blur.append(cv2.blur(x[k,:,:,:],(10,10)))
        blur=np.array(blur)
        images=Variable(torch.tensor(blur)[:,:1,:,:].cuda())
        #noise = torch.FloatTensor(images.size()).normal_(mean=0, std=25/255.0).cuda()#
        
        
        #images=images+noise
        #nse=model1(images)
        #images=images-nse#clearing out the noise
        target = Variable(im1[:,:1,:,:].cuda())
        print(iter)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if iter % 10 == 0:
            psr=psnr(outputs,target)
            print('Iteration: {}. Loss: {}. PSNR{}'.format(iter, loss.data[0],psr))
            l.append(psr)
            itr.append(iter)
            plt.plot(itr,l)
            plt.ylabel('PSNR')
            plt.xlabel('iterations')
        if(loss.data[0]<best_loss):
                best_loss=loss.data[0]
                print('saving model best10.pkl')
                torch.save(model, 'best10.pkl')
        iter=iter+1