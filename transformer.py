import torch
import torch.nn as nn

class attention_head(torch.nn.Module):
    def __init__(self,
                 embed_size,
                 head_size,
                 block_size,
                 dropout=0.2,):
        '''
        embed_size = input size (embedding size for first head)
        head_size = attention head size

        
        '''
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x = data of shape B x T x C torch tensor

        positional information is assumed to be already in x
        '''
        B, T, C = x.shape
        v = self.value(x) #B x T x hs
        k = self.key(x)#B x T x hs
        q = self.query(x)#B x T x hs

        affinities = q @ k.transpose(-2,-1)#B x T x T
        affinities *= self.head_size ** -0.5 #scale by headsize to make variance close to 1
                                            #prevent peakiness from softmax function later
         #this is for decoder blocks. if you want all to talk. delete mask
        wei = nn.functional.softmax(affinities.masked_fill(self.tril[:T,:T]==0, float('-inf')),dim=1)
        wei = self.dropout(wei)
        #wei is B x T x T. V is B x T x hs
        return wei @ v #this is B x T x hs

class multi_head_attention(nn.Module):
    '''
    multiple attention heads run in parallel
    '''
    def __init__(self,
                 embed_size,
                 head_size,
                 num_heads,
                 block_size,
                 dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([attention_head(embed_size,head_size,block_size,dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size,embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out)) #connect back to residual pathway
        return out
    
class MLP(nn.Module):
    def __init__(self,
                 embed_size,
                 dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embed_size,4*embed_size),
            nn.ReLU(),
            nn.Linear(4*embed_size,embed_size),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.model(x)
    
class Block(nn.Module):
    def __init__(self,
                 embed_size,
                 num_heads,
                 block_size,
                 dropout,):
        super().__init__()
        head_size = embed_size // num_heads
        self.attention_head = multi_head_attention(embed_size,
                                                    head_size,
                                                    num_heads,
                                                    block_size,
                                                    dropout)
        self.ff_net = MLP(embed_size,dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
    
    def forward(self,x):
        #introduces skip connections for each block
        #introduces pre-norm normalization (normalize before each operation)
        out = x + self.attention_head(self.ln1(x))
        return x + self.ff_net(self.ln2(out))