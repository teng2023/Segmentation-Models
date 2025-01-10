import torch
import torch.nn as nn

class Double_Conv(nn.Module):
    def __init__(self,first,in_ch,sample):
        super().__init__()
        if sample=='down':
            if first:
                self.double_conv=nn.Sequential(nn.Conv2d(1,in_ch,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True),
                                            nn.Conv2d(in_ch,in_ch,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(in_ch),nn.ReLU(inplace=True))
            else:
                self.double_conv=nn.Sequential(nn.Conv2d(in_ch,in_ch*2,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(in_ch*2),nn.ReLU(inplace=True),
                                            nn.Conv2d(in_ch*2,in_ch*2,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(in_ch*2),nn.ReLU(inplace=True))
        elif sample=='up':
            self.c=int(in_ch/2)
            self.double_conv=nn.Sequential(nn.Conv2d(in_ch+self.c,self.c,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(self.c),nn.ReLU(inplace=True),
                                           nn.Conv2d(self.c,self.c,kernel_size=3,stride=1,padding=0),nn.BatchNorm2d(self.c),nn.ReLU(inplace=True))
    
    def forward(self,x):
        x=self.double_conv(x)

        return x

class UNet(nn.Module):  #using upsample instead of deconvolution
    def __init__(self,num_class):
        super().__init__()
        self.down_conv1=Double_Conv(first=True,in_ch=64,sample='down')
        self.down_conv2=Double_Conv(first=False,in_ch=64,sample='down')
        self.down_conv3=Double_Conv(first=False,in_ch=128,sample='down')
        self.down_conv4=Double_Conv(first=False,in_ch=256,sample='down')
        self.down_conv5=Double_Conv(first=False,in_ch=512,sample='down')
        self.pooling=nn.MaxPool2d(kernel_size=2,stride=2)
        self.upsample=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv4=Double_Conv(first=False,in_ch=1024,sample='up')
        self.up_conv3=Double_Conv(first=False,in_ch=512,sample='up')
        self.up_conv2=Double_Conv(first=False,in_ch=256,sample='up')
        self.up_conv1=Double_Conv(first=False,in_ch=128,sample='up')
        self.final_conv=nn.Conv2d(64,num_class,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        output={}
        #down sampling
        x=self.down_conv1(x)
        output['x1']=x
        x=self.pooling(x)

        x=self.down_conv2(x)
        output['x2']=x
        x=self.pooling(x)

        x=self.down_conv3(x)
        output['x3']=x
        x=self.pooling(x)

        x=self.down_conv4(x)
        output['x4']=x
        x=self.pooling(x)
        
        x=self.down_conv5(x)

        #upsampling
        x=self.upsample(x)
        # print(x.shape)
        x=torch.cat((x,crop(output['x4'],x.shape)),dim=1)
        x=self.up_conv4(x)

        x=self.upsample(x)
        # print(x.shape)
        x=torch.cat((x,crop(output['x3'],x.shape)),dim=1)
        x=self.up_conv3(x)

        x=self.upsample(x)
        # print(x.shape)
        x=torch.cat((x,crop(output['x2'],x.shape)),dim=1)
        x=self.up_conv2(x)

        x=self.upsample(x)
        # print(x.shape)
        x=torch.cat((x,crop(output['x1'],x.shape)),dim=1)
        x=self.up_conv1(x)

        x=self.final_conv(x)

        return x
    
def crop(input,out_shape):
    height=int((input.shape[2]-out_shape[2])/2)
    weight=int((input.shape[3]-out_shape[3])/2)
    new_feature=input[:,:,height:height+out_shape[2],weight:weight+out_shape[3]]
    return new_feature

def get_param(model):
    total_sum=sum(p.numel() for p in model.parameters())
    trainable_sum=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total':total_sum,'trainable':trainable_sum}

if __name__=='__main__':
    # x=torch.randn(1,1,1024,2048)
    # model=UNet(num_class=19)
    # output=model(x)
    # print(output.shape)
    model=UNet(num_class=2)
    print(get_param(model))
