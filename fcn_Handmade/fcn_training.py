import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from fcn_model import VGGNet,FCN16s,FCN32s,FCN8s,FCNs
from fcn_cityscapes_loader import CityScapesDataset

from matplotlib import pyplot as plt
import numpy as np
import time 
import os
from tqdm import tqdm
#import argparse

n_class=19
batch_size=5
epochs=500
num_worker=2
lr=1e-4
momentum=0
w_decay=1e-5
step_size=50
gamma=0.5
configs="FCNtotal_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}_epoch".format(batch_size,epochs,step_size,gamma,lr,momentum,w_decay)
#print("Configs:",configs)

root_dir="fcn_Handmade/CityScapes/"
train_file=os.path.join(root_dir,"train.csv")
val_file=os.path.join(root_dir,"val.csv")

model_dir="fcn_Handmade/models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path=os.path.join(model_dir,configs)

use_gpu=torch.cuda.is_available()
num_gpu=list(range(torch.cuda.device_count()))
device=torch.device("cuda" if use_gpu else "cpu")

train_data=CityScapesDataset(csv_file=train_file,phase='train')
val_data=CityScapesDataset(csv_file=val_file,phase='val',flip_rate=0)
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=num_worker,pin_memory=use_gpu)
val_loader=DataLoader(val_data,batch_size=1,num_workers=num_worker,pin_memory=use_gpu)

vgg_model=VGGNet(requires_grad=True,remove_fc=True)
fcn_model=FCN8s(pretrained_net=vgg_model,n_class=n_class)

# gpu problem
if torch.cuda.device_count():
    #ts=time.time()
    vgg_model=vgg_model.to(device)
    fcn_model=fcn_model.to(device)
    fcn_model=nn.DataParallel(fcn_model,device_ids=num_gpu)
    #print("Finish cuda loading, time elapsed {}".format(time.time()-ts))

criterion=nn.BCEWithLogitsLoss()
optimizer=optim.RMSprop(fcn_model.parameters(),lr=lr,momentum=momentum,weight_decay=w_decay)
scheduler=lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)

score_dir=os.path.join("fcn_Handmade/scores",configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores=np.zeros((epochs,n_class))
pixel_scores=np.zeros(epochs)

now_time=time.localtime()

def train(retrain=False,begin=True):
    with open('fcn_Handmade/scores/score2.csv','a') as f:
        f.write(f'\n{now_time.tm_year}/{now_time.tm_mon}/{now_time.tm_mday} {now_time.tm_hour}:{now_time.tm_min}:{now_time.tm_sec}\n')

    for epoch in range(epochs):
        
        if begin:
            meaniou,pixel_accs,ious,val_loss=val(-1)
            with open('fcn_Handmade/scores/score2.csv','a') as f:
                f.write("Epoch 0/{}\n".format(epochs))
                f.write("validation loss: {}\n".format(val_loss))
                f.write("meanIOU: {}\n".format(meaniou))
                f.write("Pixel accuracy: {}\n".format(pixel_accs))
                f.write("IoUs: {}\n".format(ious))
            del meaniou,pixel_accs,ious
            begin=False

        if retrain:  #find the check point and contiue the training  
            check_point=f'fcn_Handmade/models/FCN8s_epoch_{epoch}_of_{epochs}.pth'
            if os.path.exists(check_point):
                continue
            else:
                #fcn_model=FCNs(pretrained_net=vgg_model,n_class=n_class)
                weight_path=f'fcn_Handmade/models/FCN8s_epoch_{epoch-1}_of_{epochs}.pth'
                weights=torch.load(weight_path)
                #if torch.cuda.device_count():
                    #fcn_model=fcn_model.to(device)
                    #fcn_model=nn.DataParallel(fcn_model)
                fcn_model.load_state_dict(weights,strict=True)
                retrain=False
        
        epoch_loss=0
        #ts=time.time()
        loop=tqdm(enumerate(train_loader), total=len(train_loader),leave=True)
        for iter,batch in loop:

            optimizer.zero_grad()
            # if use_gpu:
            #     inputs=Variable(batch['X'].cuda())
            #     labels=Variable(batch['Y'].cuda())
            # else:
            #     inputs,labels=Variable(batch['X']),Variable(batch['Y'])

            inputs=batch['X'].to(device)
            labels=batch['Y'].to(device)

            outputs=fcn_model(inputs)
            loss=criterion(outputs,labels)
            epoch_loss+=loss.item()
            loss.backward()
            optimizer.step()

            # if iter==(batch_size-1):
            #     print("epoch{}, loss: {}".format(epoch,loss.data[0]))
            del inputs,labels,loss
            torch.cuda.empty_cache()

            loop.set_description(f'Epoch [{epoch+1}/{epochs}]')
            loop.set_postfix(loss = (epoch_loss/(iter+1)))

        #print("Finish epoch {}, time elapsed {}".format(epoch,time.time()-ts))
        with open('fcn_Handmade/scores/score2.csv','a') as f:
            f.write("\nEpoch {}/{}\n".format(epoch+1,epochs))
            f.write("training loss: {}\n".format(epoch_loss/len(train_loader)))

        model_name='fcn_Handmade/models/FCN8s_epoch_{}_of_{}.pth'.format(epoch,epochs)
        torch.save(fcn_model.state_dict(),model_name)    #overlay the model
        scheduler.step()

        meaniou,pixel_accs,ious,val_loss=val(epoch)

        with open('fcn_Handmade/scores/score2.csv','a') as f:
            f.write("validation loss: {}\n".format(val_loss))
            f.write("meanIOU: {}\n".format(meaniou))
            f.write("Pixel accuracy: {}\n".format(pixel_accs))
            f.write("IoUs: {}\n".format(ious))
        del meaniou,pixel_accs,ious,val_loss

def val(epoch):
    with torch.no_grad():
        fcn_model.eval()
        total_ious=[]
        pixel_accs=[]
        loop=tqdm(enumerate(val_loader), total=len(val_loader),leave=True)
        val_loss=0
        for iter,batch in loop:
            # if use_gpu:
            #     inputs=batch['X'].to("cuda:0")
            #     #inputs=Variable(batch['X'].cuda())
            # else:
            #     #inputs=Variable(batch['X'])
            #     inputs=batch['X'].to('cpu')
            inputs=batch['X'].to(device)       
            labels=batch['Y'].to(device)

            output=fcn_model(inputs)
            loss=criterion(output,labels)
            val_loss+=loss.item()
            #output=output.data.cpu().numpy()
            #output=output.to('cpu').numpy()
            N,_,h,w=output.shape
            #pred=output.transpose(0,2,3,1).reshape(-1,n_class).argmax(axis=1).reshape(N,h,w)
            pred=output.permute(0,2,3,1).reshape(-1,n_class).argmax(axis=1).reshape(N,h,w)      #pred.shape=[1,1024,2048]

            #target=batch['l'].cpu().numpy().reshape(N,h,w)
            #target=batch['l'].to('cpu').numpy().reshape(N,h,w)
            target=batch['l'].reshape(N,h,w).to(device)
            for p,t in zip(pred,target):    #zip two images which channel is 1
                total_ious.append(iou(p,t)) #add a list with num_class numbers
                pixel_accs.append(pixel_acc(p,t))

            loop.set_description(f'Val [{epoch+1}/{epochs}]')
            loop.set_postfix(loss = (val_loss/(iter+1)))

            del pred,inputs,output,loss
            torch.cuda.empty_cache()

        #total_ious=total_ious.cpu().numpy()
        total_ious=torch.tensor(total_ious,device='cpu')
        pixel_accs=torch.tensor(pixel_accs,device='cpu')
        total_ious=np.array(total_ious).T   #reverse the sequence of the axis of the shape
        ious=np.nanmean(total_ious,axis=1)
        pixel_accs=np.array(pixel_accs).mean()  #shape is the number of epochs
        meaniou=np.nanmean(ious)
        print("epochs{}, loss: {}, pix_accs: {}, meanIOU: {}, IoUs: {}".format(epoch,(val_loss/len(val_loader)),pixel_accs,meaniou,ious))
        
        return meaniou,pixel_accs,ious,val_loss/len(val_loader)
        # IU_scores[epoch]=ious
        # np.save(os.path.join(score_dir,"meanIOU"),IU_scores)
        # pixel_scores[epoch]=pixel_accs
        # np.save(os.path.join(score_dir,"meanPixel"),pixel_scores)

def iou(pred,target):
    ious=[]
    for cls in range(n_class):
        pred_inds=pred==cls
        target_inds=target==cls
        intersection=pred_inds[target_inds].sum()
        union=pred_inds.sum()+target_inds.sum()-intersection
        if union==0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection)/union)
    return ious

def pixel_acc(pred,target):     #the accuracy of whole image's pixels
    correct=(pred==target).sum()
    total=(target==target).sum()
    return correct/total

if __name__=="__main__":
    #val(-1)
    #train()
    #train(retrain=True,begin=False)
    train(retrain=False,begin=True)







# class FCNTraining:
#     def __init__(self,sys_argv=None):
#         if sys_argv is None:
#             sys_argv=sys.argv[1:]
        
#         parser=argparse.ArgumentParser()
#         parser.add_argument('--n-class',help='The number of the class',default=20,type=int)
#         parser.add_argument('--batch-size',default=6)
#         parser.add_argument('--epochs',default=500)
#         parser.add_argument('lreaning-rate',default=1e-4)