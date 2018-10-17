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
                nn.ReLU(inplace=True))#0.2
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
#Loading the input image(and degradingit with noise) and target image(only noise) cuz we are doing residual learning
import torchvision.datasets as dset
import torchvision.transforms as transforms
train_image_folder = dset.ImageFolder(root='train',transform=transforms.ToTensor())
train_test_image_folder=dset.ImageFolder(root='test', transform=transforms.Compose([transforms.Resize((180,180))]))
# you can add other transformations in this list

from torch.utils.data import DataLoader
from torch.autograd import Variable
model=DeoisingCNN(num_channels=64,num_of_layers=15).cuda()
criterion = nn.MSELoss()
from torch import optim
optimizer = optim.Adam(model.parameters(), lr=0.00005)
#Clean images will be corrupted and model will learn the distribution of noise given the corrupted images
loader_train = DataLoader(dataset=train_image_folder,batch_size=6, shuffle=True)
#Test set of to check model performance on unknow lvl of noise
test= DataLoader(dataset=train_test_image_folder,batch_size=6, shuffle=False)
best_loss=np.inf
from skimage.measure import compare_ssim as ssim
epochs=26
iter=0
l=[]
itr=[]
import matplotlib.pyplot as plt

for epoch in range(epochs):
    for i, (im1,l1) in enumerate(loader_train):
        images = Variable(im1[:,:1,:,:].cuda())
        print(iter)
        #print(type(images.cpu().data))
        
        noise = torch.FloatTensor(images.size()).normal_(mean=0, std=25/255.0).cuda()#Scaling down noise
        #Goal is to train on specific noise variance and test on a range relatively far from the trained noise
        n_images=images+noise
        
        optimizer.zero_grad()
        outputs = model(n_images)
        loss = criterion(outputs, noise)
        loss.backward()
        optimizer.step()
        if iter % 5 == 0:
            print('Iteration: {}. Loss: {}. SSID: {}'.format(iter, loss.data[0],ssim(n_images[0,0,:,:].cpu().data.numpy()-outputs[0,0,:,:].cpu().data.numpy(),images[0,0,:,:].cpu().data.numpy())))
            l.append(psnr(n_images-outputs,images).cpu().item())
            itr.append(iter)
            plt.plot(itr,l)
            plt.ylabel('PSNR')
            plt.xlabel('iterations *10')
        iter=iter+1
        
        if(loss.data[0]<best_loss):
                print('saving')
                best_loss=loss.data[0]
                torch.save(model, 'noise5.pkl')

        #Evalute model performance on test set
        if (epoch%5==0):
            t=0
            for a,b in train_test_image_folder:
                x=np.asarray(a)#There is some proble in PIL to tensor conversion
                x=x[:,:,:1].reshape(1,1,180,180)#Greyscale conversion and need approriate dim of (x,1,180,180)
                #for model to work
                
                test_img=torch.from_numpy(x/255.0).float().cuda()
                std=np.random.uniform(20,30)
                nse = torch.FloatTensor(test_img.size()).normal_(mean=0, std=std/255.0).cuda()
                #torch.sum(test_img**2).cpu().item()
                nssy_img=test_img+nse
                out=model(nssy_img)
                est_image=nssy_img-out
                
                print("PSNR of test image"+str(t)+" is "+str(psnr(est_image,test_img).cpu().item()))
                t=t+1
                
