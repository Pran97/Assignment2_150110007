import torch.nn as nn
import torch
torch.cuda.empty_cache()
import numpy as np
#Model is nothing but modified version of standard VGG netowrk

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
import torchvision.datasets as dset
import torchvision.transforms as transforms
from skimage.measure import compare_ssim as ssim
train_image_folder = dset.ImageFolder(root='Set12',transform=transforms.ToTensor())
train_test_image_folder=dset.ImageFolder(root='Set12', transform=transforms.Compose([transforms.Resize((180,180))]))
t=0
from torch.utils.data import DataLoader
from torch.autograd import Variable
loader_test = DataLoader(dataset=train_test_image_folder,batch_size=12, shuffle=True)
model1=torch.load('noise3.pkl').cuda()
model2=torch.load('best7.pkl').cuda()
for a,b in train_test_image_folder:
    x=np.asarray(a).astype(float)#There is some proble in PIL to tensor conversion
    x=x/255
    
    x=x[:,:,:1].reshape(1,1,180,180)#Greyscale conversion and need approriate dim of (x,1,180,180)
    blur=cv2.blur(x[0,:,:,:],(6,6))
    
    blur=blur.reshape(1,1,180,180)
    b=x[0,0,:,:]
    b=cv2.blur(b,(6,6))#to be shown
    
    images=Variable(torch.tensor(blur)[:,:1,:,:].cuda())
    images = images.type(torch.cuda.FloatTensor)
    target=Variable(torch.tensor(x)[:,:1,:,:].cuda())
    nse = torch.FloatTensor(images.size()).normal_(mean=0, std=25/255.0).cuda()
    images=images+nse#noise addition
    out1=model1(images)
    est_img=images-out1
    out=model2(est_img)
    print('Structural Similarity Metric is '+str(ssim((out.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8),(x[0,0,:,:]*255).astype(np.uint8))))
    cv2.imshow('Original Image',(x[0,0,:,:]*255).astype(np.uint8))
    cv2.imshow('Blurred Image',(b*255).astype(np.uint8))
    cv2.imshow('Blurred Noisy Image',(images.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    cv2.imshow('Blurred Denoised Image',(est_img.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    cv2.imshow('Deblurred Denoised Image',(out.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    cv2.imwrite('im/original.png',(x[0,0,:,:]*255).astype(np.uint8))
    cv2.imwrite('im/blurred.png',(b*255).astype(np.uint8))
    cv2.imwrite('im/blurred_Noisy.png',(images.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    cv2.imwrite('im/blurred_Denoised.png',(est_img.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    cv2.imwrite('im/deblurred_Denoised.png',(out.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))

