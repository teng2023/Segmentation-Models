import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from deconvnet_model import EDeconvNet,VGG16
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import time
import numpy as np

#in order to import file from different folder
if '__file__' in globals(): 
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from fcn_Handmade.fcn_cityscapes_loader import CityScapesDataset

epochs=500
train_batch_size=1
val_batch_size=1
train_num_workers=3
val_num_workers=3
learning_rate=0.01
momentum=0.9
weight_decay=5e-4
num_class=20
step_size=20
gamma=0.5
continue_epoch=72
step_ratio=2**(continue_epoch//step_size)

now_time=time.localtime()

class_name=['background','road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus',
            'train','motorcycle','bicycle']

#create the directory for models
save_model_dir='deconvnet_handmade/check points'
if not os.path.exists(save_model_dir):
    os.makedirs(save_model_dir)

train_file='fcn_Handmade/CityScapes/train.csv'
val_file='fcn_Handmade/CityScapes/val.csv'
# test_file='fcn_Handmade/CityScapes/test.csv'
train_data=CityScapesDataset(csv_file=train_file,n_class=num_class,phase='train')
val_data=CityScapesDataset(csv_file=val_file,transform_probability=0,n_class=num_class)
# test_data=CityScapesDataset(csv_file=test_file,transform_probability=0,n_class=num_class)
train_loader=DataLoader(train_data,batch_size=train_batch_size,shuffle=True,num_workers=train_num_workers)
val_loader=DataLoader(val_data,batch_size=val_batch_size,num_workers=val_num_workers)
# test_loader=DataLoader(test_data,batch_size=1,num_workers=train_num_workers)

vgg16_model=VGG16(pretrained=True)
edeconvnet_model=EDeconvNet(pretrained_net=vgg16_model,num_class=num_class)

#GPU problem
use_gpu=torch.cuda.is_available()
device=torch.device("cuda:0" if use_gpu else "cpu")
if use_gpu:
    vgg16_model.to(device)
    edeconvnet_model.to(device)

loss_function=nn.BCEWithLogitsLoss()
# optimizer=optim.SGD(edeconvnet_model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
optimizer=optim.SGD([{'params':edeconvnet_model.parameters(),'initial_lr':learning_rate}],lr=learning_rate/step_ratio,momentum=momentum,weight_decay=weight_decay)
scheduler=lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma,last_epoch=continue_epoch)

def main(train_continue=False,begin=True):
    
    best_pixel_accu=0.7742807269096375
    best_meaniou=0.21904738247394562
    best_pixel_accu_epoch=65
    best_meaniou_epoch=62
    
    if begin:
        record_data(time=now_time,clear=True)
        val(-1)
    else:
        record_data(time=now_time,clear=False)
    
    for epoch in range(epochs):
        
        if epoch<continue_epoch:
            continue

        if train_continue:
            check_path='deconvnet_handmade/check points/EDeconvNet_Epoch{}_of_{}.pth'.format(epoch,epochs)
            if os.path.exists(check_path):
                continue
            else:
                model_keeper(epoch-1,model=edeconvnet_model,mode='reload')
                train_continue=False

        train(epoch)
        meaniou,pixel_accu=val(epoch)
        if meaniou>best_meaniou:
            best_meaniou=meaniou
            best_meaniou_epoch=epoch
        if pixel_accu>best_pixel_accu:
            best_pixel_accu=pixel_accu
            best_pixel_accu_epoch=epoch
        record_data(clear=False,best_meaniou_ep=best_meaniou_epoch,best_pixel_ep=best_pixel_accu_epoch,best_pixel=best_pixel_accu,best_meaniou=best_meaniou)

def train(epoch):
    train_tqdm=tqdm(enumerate(train_loader),total=len(train_loader),leave=True,desc=f'Train[{epoch+1}/{epochs}]')
    train_loss=0
    for i,batch in train_tqdm:
        optimizer.zero_grad()

        inputs=batch['X'].to(device)
        labels=batch['Y'].to(device)
        outputs=edeconvnet_model(inputs)
        loss=loss_function(outputs,labels)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        
        del inputs,batch,loss
        torch.cuda.empty_cache()

        train_tqdm.set_postfix(loss='{:.3f}'.format((train_loss)/(i+1)))
    
    scheduler.step()

    model_keeper(epoch=epoch,model=edeconvnet_model,mode='train')
    record_data(epoch=epoch+1,train_loss=(train_loss/len(train_loader)),clear=False,lr=scheduler.optimizer.param_groups[0]['lr'])

def val(epoch):
    with torch.no_grad():
        edeconvnet_model.eval()
        total_ious=[]
        pixel_accs=[]
        val_loss=0
        val_tqdm=tqdm(enumerate(val_loader),total=len(val_loader),leave=True,desc='Val[{}/{}]'.format(epoch+1,epochs))
        for i,batch in val_tqdm:
            
            inputs=batch['X'].to(device)
            labels=batch['Y'].to(device)

            outputs=edeconvnet_model(inputs)

            del inputs
            torch.cuda.empty_cache()

            loss=loss_function(outputs,labels)
            val_loss+=loss.item()

            del labels,loss
            torch.cuda.empty_cache()
            
            N,_,h,w=outputs.shape
            pred=outputs.permute(0,2,3,1).reshape(-1,num_class).argmax(dim=1).reshape(N,h,w)

            del outputs
            torch.cuda.empty_cache()

            target=batch['l'].reshape(N,h,w).to(device)

            total_ious.append(iou(pred,target,N))
            pixel_accs.extend(pixel_accuracy(pred,target,N))
            
            del pred,target
            torch.cuda.empty_cache()

            val_tqdm.set_postfix(loss='{:.3f}'.format((val_loss)/(i+1)))

        ious=torch.tensor(total_ious,device=device).transpose(0,1).nanmean(dim=1)
        pixel_accu=torch.tensor(pixel_accs).mean()
        meaniou=torch.nanmean(ious)
        record_data(val_loss=val_loss/len(val_loader),meaniou=meaniou,pixel_accuracy=pixel_accu,clear=False,IoUs=ious)
        return meaniou,pixel_accu

def iou(pred,target,len_batch):
    ious=[]
    for i in range(len_batch):
        for cls in range(num_class):
            pred_index=pred[i]==cls
            target_index=target[i]==cls
            intersection=pred_index[target_index].sum()
            union=pred_index.sum()+target_index.sum()-intersection
            if union==0:
                ious.append(float('nan'))
            else:
                ious.append(float(intersection)/union)
    return ious
    
def pixel_accuracy(pred,target,len_batch):
    accuracy=[]
    for i in range(len_batch):
        correct=(pred[i]==target[i]).sum()
        total=(target[i]==target[i]).sum()
        accuracy.append(correct/total)
        return accuracy

def model_keeper(epoch=None,model=None,mode=None):
    state={'model':model.state.dict(),'optimizer':optimizer.state_dict()}
    model_path='deconvnet_handmade/check points/EDeconvNet_Epoch{}_of_{}.pth'.format(epoch,epochs)
    if model and mode=='train':
        torch.save(state,model_path)
    if model and mode=='reload':
        weight=torch.load(model_path)
        model.load_state_dict(weight['model'],strict=True)
        optimizer.load_state_dict(weight['optimizer'])

def record_data(time=None,epoch=None,train_loss=None,val_loss=None,meaniou=None,pixel_accuracy=None,IoUs=None,clear=None,
                best_meaniou_ep=None,best_pixel_ep=None,lr=None,best_meaniou=None,best_pixel=None):
    data_path='deconvnet_handmade/information_20classes.csv'
    first_time=True
    if os.path.exists(data_path):
        first_time=False
    if clear:
        c='w'
    else:
        c='a'

    with open(data_path,c) as f:
        if time and first_time:
            f.write(f'{time.tm_year}/{time.tm_mon}/{time.tm_mday} {time.tm_hour}:{time.tm_min}:{time.tm_sec}\n')
        elif time:
            f.write(f'\n{time.tm_year}/{time.tm_mon}/{time.tm_mday} {time.tm_hour}:{time.tm_min}:{time.tm_sec}\n')
        if epoch:
            f.write('\nEpoch: {}/{}\n'.format(epoch,epochs))
        if train_loss:
            f.write('training loss: {}\n'.format(train_loss))
        if val_loss:
            f.write('validation loss: {}\n'.format(val_loss))
        if meaniou:
            f.write('meaniou: {}\n'.format(meaniou))
        if pixel_accuracy:
            f.write('pixel accuracy: {}\n'.format(pixel_accuracy))
        if IoUs is not None:
            f.write('IoUs: {}\n'.format(IoUs))
        if best_meaniou_ep and best_meaniou:
            f.write('best meaniou: Epoch{}\t{}\n'.format(best_meaniou_ep,best_meaniou))
        if best_pixel_ep and best_pixel:
            f.write('best pixel accuracy: Epoch{}\t{}\n'.format(best_pixel_ep,best_pixel))
        if lr:
            f.write('learning rate: {}\n'.format(lr))

def only_val():
    # for j in reversed(range(97,142)):
    for j in range(114,115):
        model_path='deconvnet_handmade/check points/EDeconvNet_Epoch{}_of_{}.pth'.format(j,epochs)
        weight=torch.load(model_path)
        edeconvnet_model.load_state_dict(weight,strict=True)
        with torch.no_grad():
            edeconvnet_model.eval()
            total_ious=[]
            pixel_accs=[]
            val_loss=0
            val_tqdm=tqdm(enumerate(val_loader),total=len(val_loader),leave=True,desc='Val[{}/{}]'.format(j+1,142))
            for i,batch in val_tqdm:
                
                inputs=batch['X'].to(device)
                labels=batch['Y'].to(device)

                outputs=edeconvnet_model(inputs)
                loss=loss_function(outputs,labels)
                val_loss+=loss.item()

                del inputs,labels,loss
                torch.cuda.empty_cache()
                
                N,_,h,w=outputs.shape
                pred=outputs.permute(0,2,3,1).reshape(-1,num_class).argmax(dim=1).reshape(N,h,w)
                target=batch['l'].reshape(N,h,w).to(device)

                total_ious.append(iou(pred,target,N))
                pixel_accs.extend(pixel_accuracy(pred,target,N))
                
                del pred,target
                torch.cuda.empty_cache()

                val_tqdm.set_postfix(loss='{:.3f}'.format((val_loss)/(i+1)))

        only_val_after(epoch=j,total_ious=total_ious,pixel_accs=pixel_accs,loss=val_loss/len(val_loader))
 
def only_val_after(epoch,total_ious,pixel_accs,loss):
    ious=torch.tensor(total_ious,device=device).transpose(0,1).nanmean(dim=1)
    pixel_accu=torch.tensor(pixel_accs).mean()
    meaniou=torch.nanmean(ious)
    record_data(epoch=epoch,val_loss=loss,meaniou=meaniou,pixel_accuracy=pixel_accu,clear=False)
    with open('deconvnet_handmade/information2.csv','a') as f:
        f.write('IoUs: {}\n'.format(ious))
        for i in range(num_class):
            f.write('{}:{}  '.format(class_name[i],ious[i]))
            if i%5==4:
                f.write('\n')

# def test():
#     with torch.no_grad():
#         edeconvnet_model.eval()
#         model_keeper(epoch=114,model=edeconvnet_model,mode='reload')
#         total_ious=[]
#         pixel_accs=[]
#         test_loss=0

#         test_tqdm=tqdm(enumerate(test_loader),total=len(test_loader),leave=True,desc='Test[1/1]')
#         for i,batch in test_tqdm:
            
#             inputs=batch['X'].to(device)
#             labels=batch['Y'].to(device)

#             outputs=edeconvnet_model(inputs)
#             loss=loss_function(outputs,labels)
#             test_loss+=loss.item()

#             del inputs,labels,loss
#             torch.cuda.empty_cache()
            
#             N,_,h,w=outputs.shape
#             pred=outputs.permute(0,2,3,1).reshape(-1,num_class).argmax(dim=1).reshape(N,h,w)

#             del outputs
#             torch.cuda.empty_cache()

#             target=batch['l'].reshape(N,h,w).to(device)

#             total_ious.append(iou(pred,target,N))
#             pixel_accs.extend(pixel_accuracy(pred,target,N))
            
#             del pred,target
#             torch.cuda.empty_cache()

#             test_tqdm.set_postfix(loss='{:.3f}'.format((test_loss)/(i+1)))

#         only_after_test(total_ious=total_ious,pixel_accs=pixel_accs,loss=test_loss/len(val_loader))

# def only_after_test(total_ious,pixel_accs,loss):
#     ious=torch.tensor(total_ious,device=device).transpose(0,1).nanmean(dim=1)
#     pixel_accu=torch.tensor(pixel_accs).mean()
#     meaniou=torch.nanmean(ious)
#     with open('deconvnet_handmade/information2.csv','a') as f:
#         f.write('\ntest\n')

#     record_data(val_loss=loss,meaniou=meaniou,pixel_accuracy=pixel_accu,clear=False)
#     with open('deconvnet_handmade/information2.csv','a') as f:
#         f.write('IoUs: {}\n'.format(ious))
#         for i in range(num_class):
#             f.write('{}:{}  '.format(class_name[i],ious[i]))
#             if i%5==4:
#                 f.write('\n')


if __name__=='__main__':
    # main(train_continue=False,begin=True)  #start from beginning
    main(train_continue=True,begin=False)
    # only_val()
    # test()