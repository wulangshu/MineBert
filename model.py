import torch.nn.functional as F
import torch
import torch.nn as nn
from config import W2VConfig
from config import AlbertConfig
from layers import *

#待改：注意力中softmax前的相似度并不是完全的余弦相似，分母是根号下hidden_embedding维度的
class word2vec(nn.Module):
    def __init__(self,W2Vconfig:W2VConfig) -> None:
        super(word2vec,self).__init__()
        self.embedding_dim=W2Vconfig.embedding_dim
        self.Qembedding=nn.Linear(in_features=W2Vconfig.vocab_dim,out_features=W2Vconfig.embedding_dim,device='cuda',dtype=torch.float32)
        self.Wembedding=nn.Linear(in_features=W2Vconfig.embedding_dim,out_features=W2Vconfig.vocab_dim,device='cuda',dtype=torch.float32)
        self.dropout=nn.Dropout(W2Vconfig.dropout_rate)

        

    def forward(self,text_onehot,length):
            
        x=self.Qembedding(text_onehot)
        x=self.dropout(x)
        hiddenx=torch.zeros(self.embedding_dim,dtype=torch.float32,device='cuda')
        for i in x:       
            hiddenx+=i
        hiddenx/=length
        outputx=self.Wembedding(hiddenx)
        outputx=F.log_softmax(outputx)
        return outputx
    

class Albert(nn.Module):
    def layernorm(self,x):
        eps: float = 0.00001
        mean = torch.mean(x[:, :], dim=(-1), keepdim=True)
        var = torch.square(x[ :, :] - mean).mean(dim=(-1), keepdim=True)
        return (x[:, :] - mean) / torch.sqrt(var + eps)#这是一般的通用实践，因为方差的无偏估计应该是除以n-1得来的，所以这里给个极小的eps作为补偿项

    def __init__(self,config:AlbertConfig):
        super(Albert,self).__init__()
        
        self.MultiHeadAttention=Multihead(config)
        self.Feedfoward1=nn.Linear(config.embedding_dim,config.feedforward_dim,device=config.device,dtype=torch.float32)
        self.Feedfoward2=nn.Linear(config.feedforward_dim,config.embedding_dim,device=config.device,dtype=torch.float32)
        
        
    def forward(self,texts_static,device):
        texts_embedding=torch.tensor([],device=device)
        for text_static in texts_static:
            x=text_static
            x=x+self.MultiHeadAttention(x,device)
            x=self.layernorm(x)
            x=self.Feedfoward1(x)
            x=F.relu(x)
            x=self.Feedfoward2(x)
            texts_embedding=torch.cat((texts_embedding,x.reshape((1,x.shape[0],x.shape[1]))),dim=0)
        return texts_embedding

class Albertwithsoftmax(nn.Module):
    def __init__(self,config:AlbertConfig,StaticEmbedding:nn.Linear):
        super(Albertwithsoftmax,self).__init__()
        self.StaticEmbedding=StaticEmbedding

        #位置编码
        self.positional_encoding=torch.tensor([i for i in range(config.txt_maxlength+1)],device=config.device,dtype=torch.float32)
        self.positional_encoding=self.positional_encoding.reshape(self.positional_encoding.shape[0],1)
        self.positional_encoding=self.positional_encoding/torch.pow(torch.tensor([10000]*config.embedding_dim,device=config.device,dtype=torch.float32),torch.tensor([2 * i / config.embedding_dim for i in range(config.embedding_dim)],device=config.device,dtype=torch.float32))
        self.positional_encoding[1:, 0::2] = torch.sin(self.positional_encoding[1:, 0::2])  # dim 2i 偶数
        self.positional_encoding[1:, 1::2] = torch.cos(self.positional_encoding[1:, 1::2])  # dim 2i+1 奇数

        self.albert=Albert(config)
        self.Wembedding=nn.Linear(in_features=config.embedding_dim,out_features=config.vocab_dim,device=config.device,dtype=torch.float32)

    def forward(self,texts_onehot,idx,device):
        texts_static=torch.tensor([],device=device)
        for i in texts_onehot:
            x=i
            x=self.StaticEmbedding(x)
            x=x+self.positional_encoding
            texts_static=torch.cat((texts_static,x.reshape((1,x.shape[0],x.shape[1]))),dim=0)
        
        x=texts_static
        for i in range(12):
            x=self.albert(x,device)#一个self-attention层循环12次
        output=torch.tensor([],device=device)
        for i in range(len(x)):
            for j in idx[i]:
                if j==-1:
                    break
                y=F.log_softmax(self.Wembedding(x[i][j]))
                output=torch.cat((output,y.reshape((1,y.shape[0]))),dim=0)
        return output