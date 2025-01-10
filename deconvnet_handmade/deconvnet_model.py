import torch
import torch.nn as nn
import torch.optim as opitm
import torchvision.models as models
from torchvision.models.vgg import VGG

# cfg={'vgg11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
#      'vgg13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
#      'vgg16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
#      'vgg19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
#      }

# ranges = {
#     'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
#     'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
#     'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
#     'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
# }

# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2,return_indices=True)]   #using return_indices will pop the error
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)

# class VGGNet(VGG):
#     def __init__(self,pretrained=True,model='vgg16',requires_grad=True,remove_fc=False,show_params=False):
#         super().__init__(vgg16_layer)
#         self.ranges=ranges[model]
#         self.remove_fc=remove_fc
#         if pretrained:
#             exec("self.load_state_dict(models.%s(weights='DEFAULT').state_dict(),strict=False)" % model)
#             # self.load_state_dict(models.vgg16(weights='DEFAULT').state_dict())

#         if not requires_grad:
#             for param in super().parameters():
#                 param.requires_grad = False

#         # if self.remove_fc:  # delete redundant fully-connected layer params, can save memory
#         #     del self.classifier

#         # self.classifier=nn.Sequential(nn.Linear(512*32*64,4096),nn.ReLU(True),nn.Dropout(p=0.5),nn.Linear(4096,4096),nn.ReLU(True),nn.Dropout(p=0.5),nn.Linear(4096,19))

#     def forward(self, x):
#         output = {}
#         all_pooling_indices=[]
#         # get the output of each maxpooling layer (5 maxpool in VGG net)
#         for idx in range(len(self.ranges)):
#             for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
#                 if layer==(self.ranges[idx][1]-1):
#                     x,pooling_index=self.features[layer](x)
#                     all_pooling_indices.append(pooling_index)
#                     print(x.shape)
#                 else:
#                     x = self.features[layer](x)
#             output["x%d"%(idx+1)] = x

#         # if not self.remove_fc:
#         #     x=output['x5']
#         #     x=torch.flatten(output, 1)
#         #     output=self.classifier(x)

#         return output,all_pooling_indices

# to suit the pretrained net in PyTorch
# vgg16_layer=nn.Sequential(nn.Conv2d(3,64,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.Conv2d(64,64,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),
#                           nn.Conv2d(64,128,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.Conv2d(128,128,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),
#                           nn.Conv2d(128,256,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.Conv2d(256,256,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.Conv2d(256,256,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),
#                           nn.Conv2d(256,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),
#                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                           nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True))

class VGG16(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,64,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.Conv2d(64,64,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),
                            nn.Conv2d(64,128,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.Conv2d(128,128,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),
                            nn.Conv2d(128,256,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.Conv2d(256,256,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.Conv2d(256,256,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),
                            nn.Conv2d(256,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True),
                            nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True))
        
        self.fc6and7_classifier=nn.Sequential(nn.Conv2d(512,4096,kernel_size=7,stride=1),nn.ReLU(inplace=True),
                                           nn.Conv2d(4096,4096,kernel_size=1,stride=1),nn.ReLU(inplace=True))
        self.maxpool_pos={4,9,16,23,30}

        if pretrained:
            self.load_state_dict(models.vgg16(weights='DEFAULT').state_dict(),strict=False)

    def forward(self,x):
        pooling_index=[]
        output={}
        number=1
        for i in range(31):
            if i in self.maxpool_pos:
                x,index=self.features[i](x)
                pooling_index.append(index)
                output[f'x{number}']=x
                number+=1
            else:
                x=self.features[i](x)
        output['x6']=self.fc6and7_classifier(x)
        return output,pooling_index

# class VGG16(VGG):
#     def __init__(self,pretrained=True):
#         super().__init__(vgg16_layer)
#         if pretrained:
#             self.load_state_dict(models.vgg16(weights='DEFAULT').state_dict())

#         self.conv1=nn.Sequential(nn.Conv2d(3,64,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                                  nn.Conv2d(64,64,kernel_size=3,padding=1),nn.ReLU(inplace=True))
#         self.pool=nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
#         self.conv2=nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                                  nn.Conv2d(128,128,kernel_size=3,padding=1),nn.ReLU(inplace=True))
#         self.conv3=nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                                  nn.Conv2d(256,256,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                                  nn.Conv2d(256,256,kernel_size=3,padding=1),nn.ReLU(inplace=True))
#         self.conv4=nn.Sequential(nn.Conv2d(256,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                                  nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                                  nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True))
#         self.conv5=nn.Sequential(nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                                  nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True),
#                                  nn.Conv2d(512,512,kernel_size=3,padding=1),nn.ReLU(inplace=True))
#         # self.classifier=nn.Sequential(nn.Linear(256,4096),nn.ReLU(inplace=True),nn.Dropout(), #let input size=(1024,2048)/5=(32,64)
#         #                               nn.Linear(4096,4096),nn.ReLU(inplace=True),nn.Dropout(),
#         #                               nn.Linear(4096,n_class))  
#         self.fc6and7_to_conv=nn.Sequential(nn.Conv2d(512,4096,kernel_size=7,stride=1),nn.ReLU(inplace=True),
#                                            nn.Conv2d(4096,4096,kernel_size=1,stride=1),nn.ReLU(inplace=True))


#     def forward(self,x):
#         all_pooling_indices=[]
#         FCN_image={}
#         output=x

#         output=self.conv1(output)

#         output,pooling_index=self.pool(output)
#         all_pooling_indices.append(pooling_index)
#         output=self.conv2(output)
#         FCN_image['x1']=output

#         output,pooling_index=self.pool(output)
#         all_pooling_indices.append(pooling_index)
#         output=self.conv3(output)
#         FCN_image['x2']=output

#         output,pooling_index=self.pool(output)
#         all_pooling_indices.append(pooling_index)
#         output=self.conv4(output)
#         FCN_image['x3']=output

#         output,pooling_index=self.pool(output)
#         all_pooling_indices.append(pooling_index)
#         output=self.conv5(output)
#         FCN_image['x4']=output

#         output,pooling_index=self.pool(output)
#         all_pooling_indices.append(pooling_index)

#         # output=self.classifier(output)
#         output=self.fc6and7_to_conv(output)
#         FCN_image['x5']=output

#         return FCN_image,all_pooling_indices

class DeconvNet(nn.Module):
    def __init__(self,pretrained_net,num_class,init_weights=True):
        super().__init__()
        self.pretrained_net=pretrained_net  #1x1x4096
        self.deconv_fc6=nn.Sequential(nn.ConvTranspose2d(4096,512,kernel_size=7,stride=1,padding=0),nn.BatchNorm2d(512),nn.ReLU(inplace=True))
        # self.unpool5=nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.deconv5=nn.Sequential(nn.ConvTranspose2d(512,512,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(512,512,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(512,512,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        # self.unpool4=nn.MaxUnpool2d(kernel_size=2,stride=2)
        self.deconv4=nn.Sequential(nn.ConvTranspose2d(512,512,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(512,512,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(512,256,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        # self.unpool3=nn.MaxUnpool2d(kernel_size=2,stride=2)                      
        self.deconv3=nn.Sequential(nn.ConvTranspose2d(256,256,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(256,256,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(256,128,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        # self.unpool2=nn.MaxUnpool2d(kernel_size=2,stride=2)                         
        self.deconv2=nn.Sequential(nn.ConvTranspose2d(128,128,kernel_size=3,stride=1,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(128,64,kernel_size=3,stride=1,padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.unpool=nn.MaxUnpool2d(kernel_size=2,stride=2)                         
        self.deconv1=nn.Sequential(nn.ConvTranspose2d(64,64,kernel_size=3,stride=1,padding=1),  nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                                   nn.ConvTranspose2d(64,64,kernel_size=3,stride=1,padding=1),  nn.BatchNorm2d(64),nn.ReLU(inplace=True),   
                                   nn.Conv2d(64,num_class,kernel_size=1,stride=1))  #remove padding=1 to fit the input size, so called classifier
        
        if init_weights:
            self.weights_initialize()                                
    
    def forward(self,x):
        output_dict,all_pooling_indices=self.pretrained_net(x)   #pretrained_net=VGG16
        output=output_dict['x6']
        output=self.deconv_fc6(output)
        output=self.unpool(output,all_pooling_indices[4])
        output=self.deconv5(output)
        output=self.unpool(output,all_pooling_indices[3])
        output=self.deconv4(output)
        output=self.unpool(output,all_pooling_indices[2])
        output=self.deconv3(output)
        output=self.unpool(output,all_pooling_indices[1])
        output=self.deconv2(output)
        output=self.unpool(output,all_pooling_indices[0])
        output=self.deconv1(output)

        return output
    
    def weights_initialize(self):
        initial_target=[self.deconv_fc6,self.deconv5,self.deconv4,self.deconv3,self.deconv2,self.deconv1]#start from upsampling layers, not from fully connected layers
        for layer in initial_target:
            for module in layer:
                if isinstance(module,nn.BatchNorm2d):
                    nn.init.constant_(module.weight,1)
                    nn.init.constant_(module.bias,0)
                elif isinstance(module,nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(module.weight,mode='fan_out',nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias,0)

class EDeconvNet(DeconvNet):    #FCNs + DeconvNet
    def __init__(self,pretrained_net,num_class,init_weights=True):
        super().__init__(pretrained_net=pretrained_net,num_class=num_class,init_weights=init_weights)
    def forward(self,x):
        output_dict,all_pooling_indices=self.pretrained_net(x)
        input=output_dict['x6']
        x5=output_dict['x5']
        x4=output_dict['x4']
        x3=output_dict['x3']
        x2=output_dict['x2']
        x1=output_dict['x1']
        output=self.deconv_fc6(input)
        output=output+x5
        output=self.unpool(output,all_pooling_indices[4])
        output=self.deconv5(output)
        output=output+x4
        output=self.unpool(output,all_pooling_indices[3])
        output=self.deconv4(output)
        output=output+x3
        output=self.unpool(output,all_pooling_indices[2])
        output=self.deconv3(output)
        output=output+x2
        output=self.unpool(output,all_pooling_indices[1])
        output=self.deconv2(output)
        output=output+x1
        output=self.unpool(output,all_pooling_indices[0])
        output=self.deconv1(output)

        return output

def get_param(model):
    total_sum=sum(p.numel() for p in model.parameters())
    trainable_sum=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total':total_sum,'trainable':trainable_sum}

if __name__=='__main__':
    # import cv2
    # image_path='fcn_Handmade/CityScapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png'
    # img=cv2.imread(image_path)
    # print(img.shape)
    # img=torch.tensor(img,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    # print(img.shape)
    # vgg16_model=VGG16(pretrained=True)
    # edeconvnet_model=EDeconvNet(pretrained_net=vgg16_model,num_class=20)
    # output=edeconvnet_model(img)
    # print(output.shape)

    vgg=VGG16(pretrained=True)
    model=EDeconvNet(vgg,num_class=2)
    print(get_param(vgg))
    print(get_param(model))
