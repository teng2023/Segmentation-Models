if '__file__' in globals(): 
    import sys,os
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from fcn_Handmade.fcn_cityscapes_loader import CityScapesDataset
from unet_model import UNet
from torchvision import transforms
import time

model_dir='unet_handmade/check points'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
train_file='fcn_Handmade/CityScapes/train.csv'
val_file='fcn_Handmade/CityScapes/val.csv'

##########################################
#initialize 
total_epochs=500
batch_size=1
num_worker=1
num_class=19
dataset=CityScapesDataset()
loss_function=nn.BCEWithLogitsLoss()
optimizer=optim.SGD()
scheduler=lr_scheduler.StepLR()
model=UNet()
learning_rate=0.01
momentum=0.9
weight_decay=5e-4
step_size=30
gamma=0.5
##########################################
class TrainingUNet():
    def __init__(self,begin=True,retrain=False,dataset=dataset,loss_function=loss_function,optimizer=optimizer,scheduler=scheduler,model=model):
        
        # training condition
        self.begin=begin
        self.retrain=retrain
       
       #dataset and dataloader
        self.train_dataset=dataset(csv_file=train_file,phase='train',n_class=num_class)
        self.val_dataset=dataset(csv_file=val_file,phase='val',flip_rate=0,n_class=num_class)
        self.train_loader=DataLoader(self.train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_worker)
        self.val_loader=DataLoader(self.val_dataset,batch_size=1,num_workers=num_worker)
        self.train_tqdm=tqdm(enumerate(self.train_loader),total=len(self.train_loader),leave=True)
        self.val_tqdm=tqdm(enumerate(self.val_loader),total=len(self.val_loader),leave=True)

        #loss function, optimizer, scheduler
        self.loss_function=loss_function
        self.optimizer=optimizer(self.model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
        self.scheduler=scheduler(self.optimizer,step_size=step_size,gamma=gamma)
        
        #model
        self.model=model(num_class=num_class)

        #file path
        self.info_path='unet_handmade/information.csv'

    def gpu_problem(self):
        self.use_gpu=torch.cuda.is_available()
        self.device=torch.device("cuda:0" if self.use_gpu else "cpu")
        if self.use_gpu:
            self.model.to(self.device)
    
    def main(self):
        self.gpu_problem()
        self.record_data(clean=False)

        if self.begin:
            self.val(epoch=epoch)

        for epoch in range(1,total_epochs+1):
            if self.retrain:
                if not self.model_keeper(epoch,mode='exist'):
                    self.model_keeper(epoch=epoch-1,mode='load')
                    self.retrain=False
                continue
            self.train(epoch=epoch)
            self.val(epoch=epoch)

    def train(self,epoch=None):
        train_loss=0
        for i,batch in self.train_tqdm:
            self.optimizer.zero_grad()

            inputs=batch['X'].to(self.device)
            labels=batch['Y'].to(self.device)

            outputs=self.model(inputs)
            del inputs
            torch.cuda.empty_cache()
            
            outputs=transforms.Resize(size=(labels.shape[0],labels.shape[1]))
            loss=self.loss_function(outputs,labels)
            train_loss+=loss.item()
            loss.backward()
            self.optimizer.step()

            del labels,loss
            torch.cuda.empty_cache()

            self.train_tqdm.set_description(f'Train Epoch{epoch}/{total_epochs}')
            self.train_tqdm.set_postfix(loss=train_loss/(i+1))
        
        self.scheduler.step()

        self.model_keeper(epoch=epoch,mode='save')
        self.record_data(epoch=epoch,train_loss=(train_loss/len(self.train_loader)))

    def val(self,epoch=None):
        with torch.no_grad():
            self.model.eval()
            total_ious=[]
            pixel_accs=[]
            val_loss=0

            for i,batch in self.val_tqdm:
                inputs=batch['X'].to(self.device)
                labels=batch['Y'].to(self.device)

                outputs=self.model(inputs)
                outputs=transforms.Resize(size=(labels.shape[0],labels.shape[1]))
                loss=self.loss_function(outputs,labels)
                val_loss+=loss.item()

                N,_,h,w=outputs.shape
                pred=outputs.permute(0,2,3,1).reshape(-1,num_class).argmax(dim=1).reshape(N,h,w)

                del inputs,labels,outputs
                torch.cuda.empty_cache()

                target=batch['l'].reshape(N,h,w).to(self.device)
                total_ious.append(self.iou(pred,target,N))
                pixel_accs.append(self.pixel_accuracy(pred,target,N))

                del pred,target
                torch.cuda.empty_cache()

                self.val_tqdm.set_description(f'Val Epoch{epoch}/{total_epochs}')
                self.val_tqdm.set_postfix(loss=val_loss/(i+1))
            
            ious=torch.tensor(total_ious,device=self.device).transpose(0,1).nanmean(dim=1)
            pixel_accu=torch.tensor(pixel_accs).mean()
            meaniou=torch.nanmean(ious,dim=1)
            
            self.record_data(val_loss=val_loss/(len(self.val_loader)),meaniou=meaniou,pixel_accuracy=pixel_accu,IoUs=ious)
            
            del total_ious,pixel_accs
            torch.cuda.empty_cache() 


    def iou(self,pred,target,len_batch):
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

    def pixel_accuracy(self,pred,target,len_batch):
        accuracy=[]
        for i in range(len_batch):
            correct=(pred[i]==target[i]).sum()
            total=(target[i]==target[i]).sum()
            accuracy.append(correct/total)
            return accuracy

    def record_data(self,clean=None,epoch=None,train_loss=None,val_loss=None,meaniou=None,pixel_accuracy=None,IoUs=None):
        now_time=time.localtime()
        if clean:
            c='w'
        else:
            c='a'
        
        with open(self.info_path,c) as f:
            if epoch:
                f.write('Epoch: {}/{}\n'.format(epoch,total_epochs))
            if train_loss:
                f.write('training loss: {}\n'.format(train_loss))
            if val_loss:
                f.write('validation loss: {}\n'.format(val_loss))
            if meaniou:
                f.write('meaniou: {}\n'.format(meaniou))
            if pixel_accuracy:
                f.write('pixel accuracy: {}\n'.format(pixel_accuracy))
            if IoUs:
                f.write('IoUs: {}\n'.format(IoUs))
            f.write(f'\n{now_time.tm_year}/{now_time.tm_mon}/{now_time.tm_mday} {now_time.tm_hour}:{now_time.tm_min}:{now_time.tm_sec}\n')

    def model_keeper(self,mode=None,epoch=None):
        model_name='Epoch{}_of_{}.pth'.format(epoch,total_epochs)
        model_path=os.path.join(model_dir,model_name)
        if mode=='save':
            torch.save(self.model.state_dict(),model_path)
        elif mode=='load':
            weight=torch.load(model_path)
            self.model.load_state_dict(weight,strcit=True)
        elif mode=='exist':
            if os.path.exists(model_dir+model_name):
                return True
            return False

if __name__=='__main__':
    TrainingUNet().main()
    # TrainingUNet(begin=False,retrain=True).main()