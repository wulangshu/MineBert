import torch.nn as nn
import torch.nn.functional as F
import torch


class SelfAttention(nn.Module):
    def __init__(self,config):
        super(SelfAttention,self).__init__()
        self.Querry=nn.Linear(config.embedding_dim,config.middle_dim,device=config.device,dtype=torch.float32)
        self.Key=nn.Linear(config.embedding_dim,config.middle_dim,device=config.device,dtype=torch.float32)
        self.Value=nn.Linear(config.embedding_dim,config.middle_dim,device=config.device,dtype=torch.float32)

    
    def forward(self,text_encoded):
        Q=self.Querry(text_encoded)
        K=self.Key(text_encoded)
        V=self.Value(text_encoded)
        weight_matrix=F.softmax(torch.mm(Q,K.T)*torch.mm((1/torch.sqrt((Q*Q).sum(dim=1))).reshape(Q.shape[0],1),(1/torch.sqrt((K*K).sum(dim=1))).unsqueeze(dim=0)),dim=1)#这里是算的cosine theata
        output=torch.mm(weight_matrix,V)      
        return output
    
class Multihead(nn.Module):
    def __init__(self,config):
        super(Multihead,self).__init__()
        self.heads=[]
        for i in range(config.num_mutilhead):
            self.heads.append(SelfAttention(config))
        self.W=nn.Linear(config.middle_dim*config.num_mutilhead,config.embedding_dim,device=config.device,dtype=torch.float32)
        
    def forward(self,text_encode,device):
        output=torch.tensor([],device=device)
        for i in self.heads:
            output=torch.cat((output,i(text_encode)),dim=1)
        output=self.W(output)
        return output
        