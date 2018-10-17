import torch.nn as nn
import torch
#torch.cuda.empty_cache()
import numpy as np
import cv2
#Model is nothing but modified version of standard VGG netowrk

class DeConv(nn.Module):
    def __init__(self):
        super(DeConv, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1,49), stride=1, padding=(0,24))#Win-k+2P+1=Wout
        #self.drop=nn.Dropout2d(0.4)
        self.T = nn.Tanh()
        
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(49,1), stride=1, padding=(24,0))
        self.T2 = nn.Tanh()
        #self.drop2=nn.Dropout2d(0.4)
        
        # Linear O/p
        self.cnn3=nn.Conv2d(in_channels=32,out_channels=1,kernel_size=5,padding=2)
    
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
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1,121), stride=1, padding=(0,60))#Win-k+2P+1=Wout
        #self.drop=nn.Dropout2d(0.4)
        
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(121,1), stride=1, padding=(60,0))
        #self.drop2=nn.Dropout2d(0.4)
        
        
        
        self.cnn3=nn.Conv2d(in_channels=20,out_channels=128,kernel_size=17,padding=8)
        
        self.cnn4=nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,padding=0)
        self.cnn5=nn.Conv2d(in_channels=128,out_channels=1,kernel_size=9,padding=4)
        
    def forward(self, x):
        # Convolution 1
        
        
        out = self.cnn1(x)
        
       
        # Convolution 2 
        out = self.cnn2(out)
        
      
        out = self.cnn3(out)
        out = self.cnn4(out)
        out = self.cnn5(out)
        return out
def psnr(i1, i2):
    mse = torch.mean( (i1 - i2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

import torch
model=torch.load('best7.pkl').cuda()#best7 and best6_noisy
import cv2
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from skimage.measure import compare_ssim as ssim
train_image_folder = dset.ImageFolder(root='Set12',transform=transforms.ToTensor())
train_test_image_folder=dset.ImageFolder(root='Set12', transform=transforms.Compose([transforms.Resize((180,180))]))
t=0
from torch.utils.data import DataLoader
from torch.autograd import Variable
loader_test = DataLoader(dataset=train_test_image_folder,batch_size=12, shuffle=False)
for a,b in train_test_image_folder:
    x=np.asarray(a).astype(float)#There is some proble in PIL to tensor conversion
    x=x/255
    
    x=x[:,:,:1].reshape(1,1,180,180)#Greyscale conversion and need approriate dim of (x,1,180,180)
    blur=cv2.blur(x[0,:,:,:],(10,10))
    
    blur=blur.reshape(1,1,180,180)
    images=Variable(torch.tensor(blur)[:,:1,:,:].cuda())
    target=Variable(torch.tensor(x)[:,:1,:,:].cuda())
    images = images.type(torch.cuda.FloatTensor)
    target=target.type(torch.cuda.FloatTensor)
    b=x[0,0,:,:]
    b=cv2.blur(b,(10,10))
    output=model(images)
    x=output.cpu().data.numpy()
    y=target.cpu().data.numpy()
    
    print("PSNR of test image"+str(t)+" is "+str(psnr(output,target).cpu().item()))
    print(ssim(x[0,0,:,:],y[0,0,:,:]))
    t=t+1
    cv2.imshow('Blurred Image',(b*255).astype(np.uint8))
    cv2.imshow('Deblurred Image',(output.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    cv2.imshow('Original Image',(target.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    
    
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

