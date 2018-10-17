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
              


import torch
model=torch.load('noise3.pkl')
import cv2
import torchvision.datasets as dset
import torchvision.transforms as transforms
from skimage.measure import compare_ssim as ssim
train_image_folder = dset.ImageFolder(root='Set12',transform=transforms.ToTensor())
train_test_image_folder=dset.ImageFolder(root='Set12', transform=transforms.Compose([transforms.Resize((180,180))]))

from torch.utils.data import DataLoader
from torch.autograd import Variable
loader_test = DataLoader(dataset=train_test_image_folder,batch_size=12, shuffle=False)
t=0
for a,b in train_test_image_folder:
    x=np.asarray(a)#There is some problem in PIL to tensor conversion
    x=x[:,:,:1].reshape(1,1,180,180)#Greyscale conversion and need approriate dim of (x,1,180,180)
    #for model to work
    
    test_img=torch.from_numpy(x/255.0).float().cuda()
    std=np.random.uniform(20,30)
    nse = torch.FloatTensor(test_img.size()).normal_(mean=0, std=25/255.0).cuda()
    #torch.sum(test_img**2).cpu().item()
    nssy_img=test_img+nse
    out=model(nssy_img)
    est_image=nssy_img-out
    
    print("PSNR of test image"+str(t)+" is "+str(psnr(est_image,test_img).cpu().item()))
    print("SSID of test image"+str(t)+" is "+str(ssim(est_image.cpu().data.numpy()[0,0,:,:],test_img.cpu().data.numpy()[0,0,:,:])))
    t=t+1
    cv2.imshow('Noisy Image',(nssy_img.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    cv2.imshow('Denoised Image',(est_image.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    cv2.imshow('Original Image',(test_img.cpu().data.numpy()[0,0,:,:]*255).astype(np.uint8))
    
    cv2.waitKey(50000)
    cv2.destroyAllWindows()
