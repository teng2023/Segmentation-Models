import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from unet_model import UNet
from torch.optim import lr_scheduler
import torch.optim as optim
import os
import time
from torchvision import transforms

#in order to import file from different folder
if '__file__' in globals(): 
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from fcn_Handmade.fcn_cityscapes_loader import CityScapesDataset

epochs=500
train_batch_size=2
val_batch_size=1
train_num_workers=1
val_num_workers=1
learning_rate=0.01
momentum=0.9
weight_decay=5e-4
num_class=20
step_size=40
gamma=0.5

#################################################
#training condition
training_begin=True
training_continue=not training_begin

continue_epoch=10
#pixel accuracy
a=0 
#meaniou
b=0
#pixel accuracy epoch
c=0
#meaniou epoch
d=0
#################################################

k=continue_epoch-10
step_ratio=2**(k//step_size)

now_time=time.localtime()

class_name=['background','road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus',
            'train','motorcycle','bicycle']

#create the directory for models and scores
save_model_dir='unet_handmade/check points4'
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

unet_model=UNet(num_class=num_class,init_weight=True,upsmaple='deconv')

#GPU problem
use_gpu=torch.cuda.is_available()
device=torch.device("cuda:0" if use_gpu else "cpu")
if use_gpu:
    unet_model.to(device)

loss_function=nn.BCEWithLogitsLoss()
optimizer=optim.SGD([{'params':unet_model.parameters(),'initial_lr':learning_rate}],lr=learning_rate/step_ratio,momentum=momentum,weight_decay=weight_decay)
# optimizer=optim.SGD(unet_model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
scheduler=lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma,last_epoch=k)

def main(train_continue=False,begin=True):

    best_pixel_accu=a
    best_meaniou=b
    best_pixel_accu_epoch=c
    best_meaniou_epoch=d
    
    if begin:
        record_data(clean=True,epoch=0)
        # val(0)
    else:
        record_data(clean=False)
    
    for epoch in range(1,epochs+1):

        # if epoch<=continue_epoch:
        #     continue

        if train_continue:
            if not model_keeper(epoch=epoch,mode='exist'):
                model_keeper(epoch=epoch-1,mode='load')
                train_continue=False
                # optimizer.param_groups[0]['lr']=learning_rate/step_ratio
            else:
                continue

        record_data(clean=False,start_time=True)

        train(epoch)

        meaniou,pixel_accu=val(epoch)
        
        if meaniou>best_meaniou:
            best_meaniou=meaniou
            best_meaniou_epoch=epoch
        if pixel_accu>best_pixel_accu:
            best_pixel_accu=pixel_accu
            best_pixel_accu_epoch=epoch
        record_data(clean=False,best_meaniou_ep=best_meaniou_epoch,best_pixel_ep=best_pixel_accu_epoch,best_pixel=best_pixel_accu,best_meaniou=best_meaniou)

def train(epoch):
    unet_model.train()
    train_tqdm=tqdm(enumerate(train_loader),total=len(train_loader),leave=True,desc=f'Train[{epoch}/{epochs}]')
    train_loss=0
    record_data(epoch=epoch,lr=scheduler.optimizer.param_groups[0]['lr'])
    for i,batch in train_tqdm:
        optimizer.zero_grad()

        inputs=batch['X'].to(device)
        labels=batch['Y'].to(device)

        outputs=unet_model(inputs)

        del inputs
        torch.cuda.empty_cache()

        resize=transforms.Resize([labels.shape[2],labels.shape[3]],antialias=True)
        outputs=resize(outputs)

        loss=loss_function(outputs,labels)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        
        del batch,loss
        torch.cuda.empty_cache()

        train_tqdm.set_postfix(loss='{:.3f}'.format((train_loss)/(i+1)))
    
    scheduler.step()

    model_keeper(epoch=epoch,mode='save')
    record_data(train_loss=(train_loss/len(train_loader)),clean=False)

def val(epoch):
    with torch.no_grad():
        unet_model.eval()
        total_ious=[]
        pixel_accs=[]
        val_loss=0
        #iter=0
        val_tqdm=tqdm(enumerate(val_loader),total=len(val_loader),leave=True,desc='Val[{}/{}]'.format(epoch,epochs))
        for i,batch in val_tqdm:
            #iter=i
            
            inputs=batch['X'].to(device)
            labels=batch['Y'].to(device)

            outputs=unet_model(inputs)

            del inputs
            torch.cuda.empty_cache()

            resize=transforms.Resize([labels.shape[2],labels.shape[3]],antialias=True)
            outputs=resize(outputs)
            loss=loss_function(outputs,labels)
            val_loss+=loss.item()

            del labels,loss
            torch.cuda.empty_cache()
            
            N,_,h,w=outputs.shape
            pred=outputs.permute(0,2,3,1).reshape(-1,num_class).argmax(dim=1).reshape(N,h,w)

            del outputs
            torch.cuda.empty_cache()

            target=batch['l'].reshape(N,h,w).to(device)

            # for p,t in zip(pred,target):    #zip two images which channel is 1
            #     total_ious.append(iou(p,t)) #add a list with num_class numbers
            #     pixel_accs.append(pixel_acc(p,t))

            total_ious.append(iou(pred,target,N))
            pixel_accs.extend(pixel_accuracy(pred,target,N))
            
            del pred,target
            torch.cuda.empty_cache()

            val_tqdm.set_postfix(loss='{:.3f}'.format((val_loss)/(i+1)))

        ious=torch.tensor(total_ious,device=device).transpose(0,1).nanmean(dim=1)
        pixel_accu=torch.tensor(pixel_accs).mean()
        meaniou=torch.nanmean(ious)
        record_data(val_loss=val_loss/len(val_loader),meaniou=meaniou,pixel_accuracy=pixel_accu,clean=False,IoUs=ious)
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

def model_keeper(mode=None,epoch=None):
    state={'model':unet_model.state_dict(),'optimizer':optimizer.state_dict()}
    model_path='unet_handmade/check points4/Epoch{}_of_{}.pth'.format(epoch,epochs)
    if mode=='save':
        torch.save(state,model_path)
    elif mode=='load':
        weight=torch.load(model_path)
        unet_model.load_state_dict(weight['model'],strict=True)
        optimizer.load_state_dict(weight['optimizer'])
    elif mode=='exist':
        if os.path.exists(model_path):
            return True
        return False

def record_data(clean=None,epoch=None,train_loss=None,val_loss=None,meaniou=None,pixel_accuracy=None,IoUs=None,
                best_meaniou_ep=None,best_pixel_ep=None,start_time=None,lr=None,best_meaniou=None,best_pixel=None):
    now_time=time.localtime()
    if clean:
        c='w'
    else:
        c='a'
    
    with open('unet_handmade/information5.csv',c) as f:
        if epoch:
            f.write('Epoch: {}/{}\n'.format(epoch,epochs))
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
        if start_time:
            f.write(f'\n{now_time.tm_year}/{now_time.tm_mon}/{now_time.tm_mday} {now_time.tm_hour}:{now_time.tm_min}:{now_time.tm_sec}\n')  

if __name__=='__main__':
    # main(train_continue=training_continue,begin=training_begin)  #start from beginning
    main(train_continue=training_continue,begin=training_begin)
    # contin=62
    # kk=contin-10
    # step_ratio=2**(kk//step_size)
    # optimizer2=optim.SGD([{'params':unet_model.parameters(),'initial_lr':learning_rate}],lr=learning_rate/step_ratio,momentum=momentum,weight_decay=weight_decay)
    # # optimizer=optim.SGD(unet_model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
    # scheduler2=lr_scheduler.StepLR(optimizer2,step_size=step_size,gamma=gamma,last_epoch=kk)
    # for i in range(0,200):
    #     print(i,optimizer2.param_groups[0]['lr'])
    #     scheduler2.step()
