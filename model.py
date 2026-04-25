import torch
import copy
from transformer import attention_head, multi_head_attention, MLP, Block

class MultiBlockModel(torch.nn.Module):
    def __init__(self,
                 num_blocks,
                 embed_size,
                 num_heads,
                 block_size,
                 vocab_size,
                 dropout,
                 device,):
        super().__init__()
        self.device = device
        self.token_table = torch.nn.Embedding(vocab_size,embed_size).to(torch.device(device))
        self.position_embedding_table = torch.nn.Embedding(block_size,embed_size).to(self.device)
        self.model = torch.nn.Sequential(
            *[Block(embed_size, num_heads, block_size, dropout) for _ in range(num_blocks)],
            torch.nn.LayerNorm(embed_size),
            torch.nn.Linear(embed_size, vocab_size)).to(device)
    
    def forward(self,x):
        B, T = x.shape
        embeddings = self.token_table(x) #(B,T,Embed)
        pos_embeddings = self.position_embedding_table(torch.arange(T,device=self.device))
        x_new = embeddings + pos_embeddings

        return self.model(x_new)
    
    def generate(self, starting_tokens,
                 max_new_tokens,
                 context_length):
        '''
        OUTDATED WITH NEW ONE
        Look up on the token_table and output the max probability of the next token
        then recursively call (or loop) on self with new last token
        '''
        all_tokens = torch.concat((torch.tensor(starting_tokens,dtype=torch.long),
                                  torch.zeros(max_new_tokens,dtype=torch.long))).to(self.device)
        window = [0, len(starting_tokens)]
        with torch.no_grad():
            while window[1] < all_tokens.shape[0]:
                latest_token = all_tokens[window[0]:window[1]].unsqueeze(0)
                logits = self(latest_token)
                token_probs = torch.nn.functional.softmax(logits[0,-1,:],
                                                          dim=0)
                next_token = torch.multinomial(token_probs, num_samples=1).item()
                #next_token = next_token = torch.argmax(token_probs).item()
                all_tokens[window[1]] = next_token
                if window[1] - window[0] == context_length:
                    window[0] += 1
                window[1] += 1
        return all_tokens.cpu().tolist()



class BigramModel(torch.nn.Module):
    def __init__(self, vocab_size,
                 block_size,
                 embed_size,
                 head_size,
                 num_heads,
                 device='cuda:1'):
        super().__init__()
        self.device=torch.device(device)
        self.token_table = torch.nn.Embedding(vocab_size,embed_size).to(torch.device(device))
        self.position_embedding_table = torch.nn.Embedding(block_size,embed_size).to(self.device)
        '''
        self.attention_head = attention_head(C=embed_size,
                                             block_size=block_size,
                                             head_size=head_size).to(self.device)
        '''
        self.attention_head = multi_head_attention(C=embed_size,
                                                   head_size = head_size // num_heads,
                                                   num_heads = num_heads,
                                                   block_size=block_size).to(self.device)
        self.ff_model = MLP(embed_size).to(self.device)
        self.head = torch.nn.Linear(head_size,vocab_size).to(torch.device(device))
        self.block_size = block_size

    def forward(self, x):
        B, T = x.shape
        embeddings = self.token_table(x) #(B,T,Embed)
        pos_embeddings = self.position_embedding_table(torch.arange(T,device=self.device))
        x_new = embeddings + pos_embeddings

        #call to attention head here, then pass this to LLM_head
        attention_out = self.attention_head(x_new)
        ff_out = self.ff_model(attention_out)
        logits = self.head(ff_out) #(B,T,C)
        return logits
    
    def generate(self, starting_tokens,
                 max_new_tokens,
                 context_length):
        '''
        OUTDATED WITH NEW ONE
        Look up on the token_table and output the max probability of the next token
        then recursively call (or loop) on self with new last token
        '''
        all_tokens = torch.concat((torch.tensor(starting_tokens,dtype=torch.long),
                                  torch.zeros(max_new_tokens,dtype=torch.long))).to(self.device)
        window = [0, len(starting_tokens)]
        with torch.no_grad():
            while window[1] < all_tokens.shape[0]:
                latest_token = all_tokens[window[0]:window[1]].unsqueeze(0)
                logits = self(latest_token)
                token_probs = torch.nn.functional.softmax(logits[0,-1,:],
                                                          dim=0)
                next_token = torch.multinomial(token_probs, num_samples=1).item()
                #next_token = next_token = torch.argmax(token_probs).item()
                all_tokens[window[1]] = next_token
                if window[1] - window[0] == context_length:
                    window[0] += 1
                window[1] += 1
        return all_tokens.cpu().tolist()

    

if __name__ == "__main__":
    pass