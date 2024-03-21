import torch
class W2VConfig():
    def __init__(self):
        self.model_name="word2vec"
        self.root_dir="D:\\mine-bert\\wiki_zh_2019\\wiki_rowlines.csv"
        self.vocab_dir='D:\\mine-bert\\wiki_zh_2019\\vocab.txt'
        self.temporary_dir='D:\\mine-bert\\wiki_zh_2019\\temporary\\'
        self.word2vec='D:\\mine-bert\\wiki_zh_2019\\word2vec.pt'
        
        self.txt_maxlength=32
        self.embedding_dim=128
        self.vocab_dim=21128
        self.dropout_rate=0.2
        self.weight_decay=1e-5
        self.batchsize=512
        self.epoch=400
        self.lr=0.01

        self.train = 0.98
        self.val = 0.02

        if torch.cuda.is_available()==True:
            self.is_cuda=True
        else:
            self.is_cuda=False

class AlbertConfig():
    def __init__(self):
        self.model_name='albert'
        self.root_dir="D:\\mine-bert\\wiki_zh_2019\\wiki_rowlines.csv"
        self.vocab_dir='D:\\mine-bert\\wiki_zh_2019\\vocab.txt'
        self.temporary_dir='D:\\mine-bert\\wiki_zh_2019\\temporary\\'
        self.word2vec='D:\\mine-bert\\wiki_zh_2019\\word2vec.pt'
        self.albert='D:\\mine-bert\\wiki_zh_2019\\albert_nodropout_69epoches.pt'
        self.staticembedding='D:\\mine-bert\\wiki_zh_2019\\staticembedding.pt'

        self.num_mutilhead=8#12
        self.txt_maxlength=32
        self.embedding_dim=128#768
        self.middle_dim=int(self.embedding_dim/self.num_mutilhead)
        self.feedforward_dim=4*self.embedding_dim
        self.vocab_dim=21128
        self.dropout_rate=0.2
        self.weight_decay=1e-5
        self.batchsize=480
        self.epoch=400
        self.lr=0.01
        #太慢了太慢了太慢了
        self.train=0.98
        self.val=0.02

        
        if torch.cuda.is_available()== True:
            self.device='cuda'
        else:
            self.device='cpu'
