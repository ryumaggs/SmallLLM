import torch
import copy
class BigramModel(torch.nn.Module):
    def __init__(self, vocab_size,
                 block_size,
                 embed_size,
                 device='cuda:1'):
        super().__init__()
        self.device=torch.device(device)
        self.token_table = torch.nn.Embedding(vocab_size,embed_size).to(torch.device(device))
        self.position_embedding_table = torch.nn.Embedding(block_size,embed_size).to(self.device)
        self.head = torch.nn.Linear(embed_size,vocab_size).to(torch.device(device))

    def forward(self, x):
        B, T = x.shape
        embeddings = self.token_table(x) #(B,T,Embed)
        pos_embeddings = self.position_embedding_table(torch.arange(T,device=self.device))
        x_new = embeddings + pos_embeddings
        logits = self.head(x_new) #(B,T,C)
        return logits
    
    def generate(self, starting_tokens,
                 max_new_tokens,):
        '''
        OUTDATED WITH NEW ONE
        Look up on the token_table and output the max probability of the next token
        then recursively call (or loop) on self with new last token
        '''

        output_tokens = copy.deepcopy(starting_tokens)
        with torch.no_grad():
            for i in range(1,max_new_tokens+1):
                latest_token = output_tokens[i-1]
                token_probs = torch.nn.functional.softmax(self.token_table.weight[latest_token],
                                                          dim=0)
                next_token = torch.multinomial(token_probs, num_samples=1).item()
                output_tokens.append(next_token)
        return output_tokens

    

if __name__ == "__main__":
    pass