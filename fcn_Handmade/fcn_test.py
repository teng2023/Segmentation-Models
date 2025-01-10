import torch
import numpy as np
from fcn_model import VGGNet,FCNs
from torchvision.io import read_image
from torchvision.transforms import v2
import cv2
import torch.nn as nn
from fcn_cityscapes_utils import indextocolor
from PIL import Image
from tqdm import tqdm
from fcn_training import iou,pixel_acc

#Not tested yet
def paint_output_image(output):     #output is a image after putting into the model
    N,n_class,h,w=output.shape
    assert N==1
    pred_img=output.squeeze(0).permute(1,2,0).reshape(-1,n_class).argmax(axis=1).reshape(h,w)
    pred_img=torch.stack((pred_img,pred_img,pred_img),dim=0)
    cap=torch.zeros_like(pred_img)
    for i in range(n_class):
        # if i==1:
        mask=(pred_img==i)*1
        color_tuple=indextocolor(i)
        mask[0]*=color_tuple[0]
        mask[1]*=color_tuple[1]
        mask[2]*=color_tuple[2]
        cap+=mask
                
    return cap   #type=numpy array, shape=(h,w,3)
    
if __name__=='__main__':
    
    model_path='fcn_Handmade/models/FCN_epoch_127_of_500.pth'
    model2_path='fcn_Handmade/models2/FCN8s_epoch_0_of_500.pth'
    test_path='fcn_Handmade/Cityscapes/leftImg8bit/test/berlin/berlin_000001_000019_leftImg8bit.png'
    test_original_path='fcn_Handmade/CityScapes/Labeled_idx/test/berlin/berlin_000000_000019_gtFine_color.png.npy'
    train_path='fcn_Handmade/Cityscapes/leftImg8bit/train/aachen/aachen_000009_000019_leftImg8bit.png'
    n_class=19
    vgg_model=VGGNet(requires_grad=True,remove_fc=True)
    fcn_model=FCNs(pretrained_net=vgg_model,n_class=n_class)
    fcn_model=nn.DataParallel(fcn_model)
    weight=torch.load(model_path)
    fcn_model.load_state_dict(weight)
    
    img=cv2.imread(test_path).astype(np.float64)
    ori_img = np.copy(img)
    means=np.array([123.68,103.939, 116.779])
    # img[0] -=means[2]
    # img[1] -=means[1]
    # img[2] -=means[0]
    img/=255
    img=torch.tensor(img,dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    

    output=fcn_model(img)
    a=output.permute(0,2,3,1).reshape(-1,n_class).argmax(axis=1).reshape(1024,2048)
    output=paint_output_image(output).cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)

    pred = cv2.addWeighted(ori_img, 0.5, output, 0.8, 0.0,dtype=cv2.CV_8UC3)

    cv2.imshow('a',pred)
    cv2.waitKey(0)
    
    # a=v2.Resize((512,1024),antialias=None)
    # img=read_image(test_path).unsqueeze(0).to(dtype=torch.float32)
    # img=a(img)
    # output=fcn_model(img)
    # a = (output[0, 14]>1)*1.0
    # a=a.to('cpu')
    # a=np.stack((a*200/255, a*1/255, a*1/255), 2)
    # b= (output[0, 5]>1.2)*1.0
    # b=b.to('cpu')
    # b=np.stack((b*2/255, b*200/255, b*1/255), 2)
    # a=np.array(a)
    # b=np.array(b)
    # cv2.imshow('asdf',b)
    # cv2.waitKey(0)

    # c=np.load('fcn_Handmade/scores/FCN-BCEWithLogits_batch5_total_epoch500_RMSprop_scheduler-step50-gamma0.5_lr0.0001_momentum0_w_decay1e-05_epoch/meanIOU.npy')
    # print(c[10])

    #reproduce the image in gtFine
    # d=np.load('fcn_Handmade/CityScapes/Labeled_idx/train/aachen/aachen_000000_000019_gtFine_color.png.npy')

    # final_img=np.zeros((1024,2048,3))

    # for i in range(1024):
    #     for j in range(2048):
    #         color_tuple=indextocolor(d[i,j])
    #         final_img[i,j,0]=color_tuple[0]
    #         final_img[i,j,1]=color_tuple[1]
    #         final_img[i,j,2]=color_tuple[2]


    # img=Image.fromarray(np.uint8(final_img))
    # img.show()
