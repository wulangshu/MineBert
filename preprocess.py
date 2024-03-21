import numpy as np
import scipy.sparse as sp
import torch
import random
import pandas as pd
from torch.utils.data import Dataset


def W2Vtokenizer(string:str,line_length,vocab_dim,vocab:list): 
    length=len(string)
    mask=random.randint(0,length-1)
    try:
        label=vocab[string[mask]]
    except:
        label=vocab['[UNK]']
    matrixeye=np.eye(vocab_dim)
    box=np.array([matrixeye[vocab['[CLS]']]])

    for i in range(length):
        if i==mask:
            box=np.append(box,[np.zeros(vocab_dim)],axis=0)
        else:
            try:
                box=np.append(box,[matrixeye[vocab[string[i]]]],axis=0)
            except:
                box=np.append(box,[matrixeye[vocab['[UNK]']]],axis=0)
    for i in range(line_length-length):
        box=np.append(box,[matrixeye[vocab['[PAD]']]],axis=0)
    box=box.astype(np.float32)
    box=sp.coo_matrix(box)
    return box,label,length

class W2Vdataset(Dataset):
    
    def __init__(self,config,is_train:bool,is_copy:bool,need_shuffle):  
        self.vocab=[]
        self.text=np.array([],dtype=np.float32)
        self.label=np.array([],dtype=np.int32)
        self.length=np.array([],dtype=np.int32)
        self.mid=np.array([],dtype=np.int32)
        self.lid=np.array([],dtype=np.int32)
        self.shuffle=None
        if is_copy==False:
            with open(config.vocab_dir,mode="r",encoding="utf-8",errors="ignore") as r:
                lines=r.readlines()
            for line in lines:
                self.vocab.append(line.strip('\n'))
            self.vocab={vocabulary:index  for index,vocabulary in enumerate(self.vocab)}

            file=pd.read_csv(config.root_dir)
            file_mid=np.array(file['mid'])
            self.mid=file_mid.copy()
            file_lid=np.array(file['lid'])
            self.lid=file_lid.copy()
            file_text=np.array(file['text'])
            lines=len(file_text)
            #shuffle
            if type(need_shuffle)==type(True) and need_shuffle==True:
                self.shuffle=np.random.permutation(lines)
            elif type(need_shuffle)==type(True) and need_shuffle==False:
                self.shuffle=np.arange(lines)
            else:
                self.shuffle=need_shuffle
            if type(need_shuffle)==type(True):
                with open(file=config.temporary_dir+'w2v_shuffle.npy',mode='wb')as f:
                    np.save(f, self.shuffle)   
            if is_train==True:
                file_text=file_text[self.shuffle[:int(lines*config.train)]]
                self.mid=self.mid[self.shuffle[:int(lines*config.train)]]
                self.lid=self.lid[self.shuffle[:int(lines*config.train)]]
            else:
                file_text=file_text[self.shuffle[int(lines*config.train):]]
                self.mid=self.mid[self.shuffle[int(lines*config.train):]]
                self.lid=self.lid[self.shuffle[int(lines*config.train):]]
            lines=len(file_text)
            count=0
            for i in range(lines):
                if (len(file_text[i])>config.txt_maxlength):
                    print(file_text[i])
                return_text,return_label,return_length=W2Vtokenizer(file_text[i].strip('\t'),config.txt_maxlength,config.vocab_dim,self.vocab)
                self.text=np.append(self.text,return_text)
                self.label=np.append(self.label,return_label)
                self.length=np.append(self.length,return_length)
                count+=1
            
            if type(need_shuffle)==type(True):
                with open(file=config.temporary_dir+'w2v_training_mid.npy',mode='wb')as f:
                    np.save(f, self.mid)    

                with open(file=config.temporary_dir+'w2v_training_lid.npy',mode='wb')as f:
                    np.save(f, self.lid)    

                with open(file=config.temporary_dir+'w2v_training_text.npy',mode='wb')as f:
                    np.save(f, self.text)    

                with open(file=config.temporary_dir+'w2v_training_label.npy',mode='wb')as f:
                    np.save(f, self.label)    

                with open(file=config.temporary_dir+'w2v_training_length.npy',mode='wb')as f:
                    np.save(f, self.length)    
            else:
                with open(file=config.temporary_dir+'w2v_val_mid.npy',mode='wb')as f:
                    np.save(f, self.mid)    

                with open(file=config.temporary_dir+'w2v_val_lid.npy',mode='wb')as f:
                    np.save(f, self.lid)    

                with open(file=config.temporary_dir+'w2v_val_text.npy',mode='wb')as f:
                    np.save(f, self.text)    

                with open(file=config.temporary_dir+'w2v_val_label.npy',mode='wb')as f:
                    np.save(f, self.label)    

                with open(file=config.temporary_dir+'w2v_val_length.npy',mode='wb')as f:
                    np.save(f, self.length) 
        else:
            if is_train == True:
                target='training'
            else:
                target='val'

            with open(config.vocab_dir,mode="r",encoding="utf-8",errors="ignore") as r:
                lines=r.readlines()
            for line in lines:
                self.vocab.append(line.strip('\n'))
            self.vocab={vocabulary:index  for index,vocabulary in enumerate(self.vocab)}
            
            
            self.shuffle= np.load(config.temporary_dir+'w2v_shuffle.npy', allow_pickle=True) 

            self.mid=np.load(config.temporary_dir+'w2v_'+target+'_mid.npy', allow_pickle=True)    

            self.lid=np.load(config.temporary_dir+'w2v_'+target+'w2v_lid.npy', allow_pickle=True)    

            self.text=np.load(config.temporary_dir+'w2v_'+target+'w2v_text.npy', allow_pickle=True)    

            self.label=np.load(config.temporary_dir+'w2v_'+target+'w2v_label.npy', allow_pickle=True)    

            self.length=np.load(config.temporary_dir+'w2v_'+target+'w2v_length.npy', allow_pickle=True) 
      
        

    def __len__(self):
        return len(self.lid)

    def __getitem__(self, index):
        return torch.from_numpy(self.mid[[index]]),torch.from_numpy(self.lid[[index]]),torch.from_numpy(self.text[index].todense()),torch.from_numpy(self.label[[index]]),torch.from_numpy(self.length[[index]])
    
def Alberttokenizer(string:str,line_length,vocab_dim,vocab:list): 
    length=len(string)
    label=[]
    string_idx=[i for i in range(0,length)]
    mask_idx=[]
    for i in range(max(1,int(length*0.15))):
        ran_num=random.randint(0,len(string_idx)-1)
        mask_idx.append(string_idx[ran_num]+1)
        try:
            label.append(vocab[string[string_idx[ran_num]]])
        except:
            label.append(vocab['[UNK]'])
    
        del string_idx[ran_num]

    for i in range(max(1,int(line_length*0.15))-max(1,int(length*0.15))):#padding
        mask_idx.append(-1)
        label.append(-1)
          
    matrixeye=np.eye(vocab_dim)
    box=np.array([matrixeye[vocab['[CLS]']]])

    for i in range(length):
        if i+1 in mask_idx:
            decission=random.random()
            if decission<0.8:
                box=np.append(box,[matrixeye[vocab['[MASK]']]],axis=0)
            elif decission>0.9:
                box=np.append(box,[matrixeye[random.randint(0,vocab_dim-1)]],axis=0)
            else:
                try:
                    box=np.append(box,[matrixeye[vocab[string[i]]]],axis=0)
                except:
                    box=np.append(box,[matrixeye[vocab['[UNK]']]],axis=0)
        else:
            try:
                box=np.append(box,[matrixeye[vocab[string[i]]]],axis=0)
            except:
                box=np.append(box,[matrixeye[vocab['[UNK]']]],axis=0)
    for i in range(line_length-length):
        box=np.append(box,[matrixeye[vocab['[PAD]']]],axis=0)
    box=box.astype(np.float32)
    box=sp.coo_matrix(box)
    return box,label,mask_idx,length

class Albertdataset():
    
    def __init__(self,config,is_train:bool,is_copy:bool,need_shuffle):  
        self.vocab=[]
        self.text=np.array([],dtype=np.float32)
        self.label=np.array([],dtype=np.int32)
        self.idx=np.array([],dtype=np.int32)
        self.length=np.array([],dtype=np.int32)
        self.mid=np.array([],dtype=np.int32)
        self.lid=np.array([],dtype=np.int32)
        self.shuffle=None
        if is_copy==False:
            with open(config.vocab_dir,mode="r",encoding="utf-8",errors="ignore") as r:
                lines=r.readlines()
            for line in lines:
                self.vocab.append(line.strip('\n'))
            self.vocab={vocabulary:index  for index,vocabulary in enumerate(self.vocab)}

            file=pd.read_csv(config.root_dir)
            file_mid=np.array(file['mid'])
            self.mid=file_mid.copy()
            file_lid=np.array(file['lid'])
            self.lid=file_lid.copy()
            file_text=np.array(file['text'])
            lines=len(file_text)
            #shuffle
            if type(need_shuffle)==type(True) and need_shuffle==True:
                self.shuffle=np.random.permutation(lines)
            elif type(need_shuffle)==type(True) and need_shuffle==False:
                self.shuffle=np.arange(lines)
            else:
                self.shuffle=need_shuffle
            if type(need_shuffle)==type(True):
                with open(file=config.temporary_dir+'albert_shuffle.npy',mode='wb')as f:
                    np.save(f, self.shuffle)   
            if is_train==True:
                file_text=file_text[self.shuffle[:int(lines*config.train)]]
                self.mid=self.mid[self.shuffle[:int(lines*config.train)]]
                self.lid=self.lid[self.shuffle[:int(lines*config.train)]]
            else:
                file_text=file_text[self.shuffle[int(lines*config.train):]]
                self.mid=self.mid[self.shuffle[int(lines*config.train):]]
                self.lid=self.lid[self.shuffle[int(lines*config.train):]]
            lines=len(file_text)
            count=0
            for i in range(lines):
                if (len(file_text[i])>config.txt_maxlength):
                    print(file_text[i])
                return_text,return_label,return_idx,return_length=Alberttokenizer(file_text[i].strip('\t'),config.txt_maxlength,config.vocab_dim,self.vocab)
                self.text=np.append(self.text,return_text)
                self.label=np.append(self.label,return_label)
                self.idx=np.append(self.idx,return_idx)
                self.length=np.append(self.length,return_length)
                count+=1
            self.label=self.label.reshape(lines,int(config.txt_maxlength*0.15))
            self.idx=self.idx.reshape(lines,int(config.txt_maxlength*0.15))

            if type(need_shuffle)==type(True):
                with open(file=config.temporary_dir+'albert_training_mid.npy',mode='wb')as f:
                    np.save(f, self.mid)    

                with open(file=config.temporary_dir+'albert_training_lid.npy',mode='wb')as f:
                    np.save(f, self.lid)    

                with open(file=config.temporary_dir+'albert_training_text.npy',mode='wb')as f:
                    np.save(f, self.text)    

                with open(file=config.temporary_dir+'albert_training_label.npy',mode='wb')as f:
                    np.save(f, self.label)    

                with open(file=config.temporary_dir+'albert_training_idx.npy',mode='wb')as f:
                    np.save(f, self.idx)

                with open(file=config.temporary_dir+'albert_training_length.npy',mode='wb')as f:
                    np.save(f, self.length)    
            else:
                with open(file=config.temporary_dir+'albert_val_mid.npy',mode='wb')as f:
                    np.save(f, self.mid)    

                with open(file=config.temporary_dir+'albert_val_lid.npy',mode='wb')as f:
                    np.save(f, self.lid)    

                with open(file=config.temporary_dir+'albert_val_text.npy',mode='wb')as f:
                    np.save(f, self.text)    

                with open(file=config.temporary_dir+'albert_val_label.npy',mode='wb')as f:
                    np.save(f, self.label)    

                with open(file=config.temporary_dir+'albert_val_idx.npy',mode='wb')as f:
                    np.save(f, self.idx)  

                with open(file=config.temporary_dir+'albert_val_length.npy',mode='wb')as f:
                    np.save(f, self.length) 
        else:
            if is_train == True:
                target='training'
            else:
                target='val'

            with open(config.vocab_dir,mode="r",encoding="utf-8",errors="ignore") as r:
                lines=r.readlines()
            for line in lines:
                self.vocab.append(line.strip('\n'))
            self.vocab={vocabulary:index  for index,vocabulary in enumerate(self.vocab)}
            
            
            self.shuffle= np.load(config.temporary_dir+'albert_shuffle.npy', allow_pickle=True) 

            self.mid=np.load(config.temporary_dir+'albert_'+target+'_mid.npy', allow_pickle=True)    

            self.lid=np.load(config.temporary_dir+'albert_'+target+'_lid.npy', allow_pickle=True)    

            self.text=np.load(config.temporary_dir+'albert_'+target+'_text.npy', allow_pickle=True)    

            self.label=np.load(config.temporary_dir+'albert_'+target+'_label.npy', allow_pickle=True) 

            self.idx=np.load(config.temporary_dir+'albert_'+target+'_idx.npy', allow_pickle=True)   

            self.length=np.load(config.temporary_dir+'albert_'+target+'_length.npy', allow_pickle=True) 
      
        

    def __len__(self):
        return len(self.lid)

    def __getitem__(self, index):
        return torch.from_numpy(self.mid[[index]]),torch.from_numpy(self.lid[[index]]),torch.sparse_coo_tensor(indices=torch.from_numpy(np.array([self.text[index].row,self.text[index].col])),values=torch.from_numpy(self.text[index].data),size=torch.Size(self.text[index].shape)),torch.from_numpy(self.label[index]),torch.from_numpy(self.idx[index]),torch.from_numpy(self.length[[index]])