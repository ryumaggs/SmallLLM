import torch
import torch.nn as nn

class attention_head(torch.nn.module):
    def __init__(self,
                 C,
                 head_size):
        '''
        C = input size (embedding size for first head)
        head_size = attention head size

        
        '''
        super.init()
        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)

    def forward(self, x):
        '''
        x = data of shape B x T x C torch tensor

        positional information is assumed to be already in x
        '''
        B, T, C = x.shape()
        v = self.value(x) #B x T x hs
        k = self.key(x)#B x T x hs
        q = self.query(x)#B x T x hs

        affinities = q @ k.transpose(-2,-1) #B x T x T
        mask = torch.tril(torch.ones(T,T)) #this is for decoder blocks. if you want all to talk. delete mask
        
        wei = nn.functional.softmax(affinities.masked_fill(mask==0,torch.float('-inf')),dim=1)
        #wei is B x T x T. V is B x T x hs
        return wei @ v #this is B x T x hs
