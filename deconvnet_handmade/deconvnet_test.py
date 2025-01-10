import torch
import os

if '__file__' in globals(): 
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from deconvnet_handmade.deconvnet_model import VGG16,EDeconvNet
import cv2
import numpy as np
from fcn_Handmade.fcn_cityscapes_loader import CityScapesDataset
from fcn_Handmade.fcn_cityscapes_utils import indextocolor


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
    model_path='deconvnet_handmade/check points/EDeconvNet_Epoch114_of_500.pth'
    test_path='fcn_Handmade/Cityscapes/leftImg8bit/test/berlin/berlin_000068_000019_leftImg8bit.png'
    # train_path='fcn_Handmade/CityScapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'

    n_class=19
    vgg_model=VGG16(pretrained=True)
    edeconvnet_model=EDeconvNet(vgg_model,num_class=n_class)
    weight=torch.load(model_path)
    edeconvnet_model.load_state_dict(weight)
    edeconvnet_model.eval()

    img=cv2.imread(test_path).astype(np.float64)
    ori_img = np.copy(img)
    # img[0]-=103.939
    # img[1]-=116.779
    # img[2]-=123.68
    img/=255
    img=torch.tensor(img,dtype=torch.float32).permute(2,0,1).unsqueeze(0)

    output=edeconvnet_model(img)
    output=paint_output_image(output).cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)

    pred = cv2.addWeighted(ori_img, 0.5, output, 0.8, 0.0,dtype=cv2.CV_8UC3)

    cv2.imshow('a',pred)
    cv2.waitKey(0)

###################################################################################
    # from torchvision import transforms

    # img/=255
    # cv2.imshow('A',img)
    # cv2.waitKey(0)
    # img=torch.from_numpy(img.copy()).float().permute(2,0,1)
    # a=transforms.Compose([transforms.RandomRotation(degrees=45)])
    # b=transforms.Compose([transforms.ColorJitter(brightness=0,contrast=0,saturation=0,hue=0)])
    # img=a(img)
    # # print(type(img))
    # print(img.shape)
    # img=np.array(img).transpose(1,2,0)
    # # print(type(img))
    # print(img.shape)
    # cv2.imshow('A',img)
    # cv2.waitKey(0)
