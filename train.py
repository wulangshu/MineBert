from preprocess import *
from torch.utils.data.dataloader import DataLoader
from config import *
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

#word2vec
"""W2Vconfig=W2VConfig()
model=torch.load(W2Vconfig.word2vec)#word2vec(W2Vconfig)
if W2Vconfig.is_cuda == True:
    model.cuda()
optimizer=optim.Adam(model.parameters(),lr=W2Vconfig.lr,weight_decay=W2Vconfig.weight_decay)
dataset_train=W2Vdataset(W2Vconfig,True,True,need_shuffle=True)
dataset_val=W2Vdataset(W2Vconfig,False,True,dataset_train.shuffle)
dataloader_train=DataLoader(dataset_train,batch_size=W2Vconfig.batchsize,num_workers=2)
dataloader_val=DataLoader(dataset_val,batch_size=W2Vconfig.batchsize,num_workers=2)"""
#3
#albert
Albertconfig=AlbertConfig()
model=torch.load(Albertconfig.albert)#Albertwithsoftmax(Albertconfig,torch.load(Albertconfig.word2vec))
if Albertconfig.device== 'cuda':
    model.cuda()
optimizer=optim.Adam(model.parameters(),lr=Albertconfig.lr,weight_decay=Albertconfig.weight_decay)
dataset_train=Albertdataset(Albertconfig,True,True,need_shuffle=True)
dataset_val=Albertdataset(Albertconfig,False,True,dataset_train.shuffle)
dataloader_train=DataLoader(dataset_train,batch_size=Albertconfig.batchsize,num_workers=0)
dataloader_val=DataLoader(dataset_val,batch_size=Albertconfig.batchsize,num_workers=0)

def W2Vfit(epoch,dataloader,is_train:bool,is_cuda):#,volatile=False):
    if is_train == True:
        model.train()
    else:
        model.eval()
        #volatile=True
    running_loss=0.0
    running_correct=0.0
    for batch_idx,(mid,lid,text,label,length) in enumerate(dataloader):
        if is_cuda==True:
            text,label,length=text.cuda(),label.cuda(),length.cuda()
        
        if is_train==True:
            optimizer.zero_grad()
        loss=torch.tensor([],device='cuda')#torch.tensor(0,dtype=torch.float32,device='cuda')
        batch_correct=0
        prediction=torch.Tensor()
        for i in range(len(text)):
            prediction=model(text[i],length[i])
            loss=torch.cat((loss,prediction.reshape((1,prediction.shape[0]))),0)#loss+=1-prediction[label[i].item()]
            #running_loss+=loss.item()
            if prediction.data.max(dim=0,keepdim=True)[1].item()==label[i]:
                batch_correct+=1
        """     
            np.append(preds,prediction.data.max(dim=0,keepdim=True)[1].cpu().numpy())
        preds=torch.from_numpy(preds)
        correct=preds.eq(label.data.view_as(preds)).cpu().sum()
        """
        running_loss=F.nll_loss(loss,label.reshape(label.shape[0]).type(torch.int64))
        running_correct+=batch_correct
        if is_train==True:
            running_loss.backward()
            optimizer.step()
        print('batch_idx',batch_idx,'   ','correct:',batch_correct)
    running_loss=running_loss.item()/len(dataloader.dataset)
    running_correct=running_correct/len(dataloader.dataset)
    if is_train== True:
        print('epoch',epoch,'train_loss:',running_loss,'   ','train_correct:',running_correct)
    else:
        print('epoch',epoch,'val_loss:',running_loss,'   ','val_correct:',running_correct)
    return running_loss,running_correct

def Albertfit(epoch,dataloader,is_train:bool,device):#,volatile=False):
    if is_train == True:
        model.train()
    else:
        model.eval()
        #volatile=True
    running_loss=0.0
    running_correct=0.0
    for batch_idx,(mid,lid,text,label,idx,length) in enumerate(dataloader):
        if device=='cuda':
            text,label,idx,length=text.cuda(),label.cuda(),idx.cuda(),length.cuda()
        
        if is_train==True:
            optimizer.zero_grad()

        label=label.flatten()
        k=[]
        for i in range(len(label)):
            if label[i].item()!=-1:
                k.append(i)
        k=torch.tensor(k)
        label=label[k]

        batch_correct=0
        prediction=model(text,idx,device) 
        running_loss=F.nll_loss(prediction,label.type(torch.int64))
           
        for i in range(len(prediction)):
            if prediction[i].data.max(dim=0,keepdim=True)[1].item()==label[i]:
                batch_correct+=1
       
        running_correct+=batch_correct
        if is_train==True:
            running_loss.backward()
            optimizer.step()
        print('batch_idx',batch_idx,'   ','correct:',batch_correct)
    running_loss=running_loss.item()/len(dataloader.dataset)
    running_correct=running_correct/len(dataloader.dataset)
    if is_train== True:
        print('epoch',epoch,'train_loss:',running_loss,'   ','train_correct:',running_correct)
    else:
        print('epoch',epoch,'val_loss:',running_loss,'   ','val_correct:',running_correct)
    return running_loss,running_correct
       
       

if __name__=='__main__':
    
    train_losses,train_accuracy=[],[]
    val_losses,val_accuracy=[],[]
    #训练static embedding
    """for epoch in range(W2Vconfig.epoch):
        print("-------现在是轮次 "+str(epoch)+"-------")
        epoch_train_losses,epoch_train_accuracy=W2Vfit(epoch,dataloader_train,True,W2Vconfig.is_cuda)
        epoch_val_losses,epoch_val_accuracy=W2Vfit(epoch,dataloader_val,False,W2Vconfig.is_cuda)
        train_losses.append(epoch_train_losses)
        train_accuracy.append(epoch_train_accuracy)
        val_losses.append(epoch_val_losses)
        val_accuracy.append(epoch_val_accuracy)
    torch.save(model.Qembedding,W2Vconfig.word2vec)"""

    #训练dynamic embedding
    for epoch in range(Albertconfig.epoch):
        print("-------现在是轮次 "+str(epoch)+"-------")
        epoch_train_losses,epoch_train_accuracy=Albertfit(epoch,dataloader_train,True,Albertconfig.device)
        epoch_val_losses,epoch_val_accuracy=Albertfit(epoch,dataloader_val,False,Albertconfig.device)
        train_losses.append(epoch_train_losses)
        train_accuracy.append(epoch_train_accuracy)
        val_losses.append(epoch_val_losses)
        val_accuracy.append(epoch_val_accuracy)
    torch.save(model.albert,Albertconfig.albert)
    torch.save(model.StaticEmbedding,Albertconfig.staticembedding)
    """plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label='trainingloss')
    plt.plot(range(1,len(val_losses)+1),val_losses,'r',label='valloss')
    plt.legend()"""
    with open ('D:\\mine-bert\\wiki_zh_2019\\train_correct.txt','w') as w:
        for i in train_accuracy:
            w.writelines(str(i)+'\n')


