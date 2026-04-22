import torch
import numpy as np

class shakeData():
    def __init__(self, 
                 path,
                 device='cuda:1'):
        with open(path, 'r') as file:
            content = file.read()
        uc = sorted(list(set(content)))
        self.vocab_size = len(uc)
        self.encoding = {uc[i]:i for i in range(len(uc))}
        self.decoding = {i: uc[i] for i in range(len(uc))}

        train_percent = 0.9
        train_cutoff = int(len(content) * 0.9)
        self.training = torch.tensor(np.array(self.encode(content[:train_cutoff])),dtype=torch.long, device=device)
        self.validation = torch.tensor(np.array(self.encode(content[train_cutoff:])),dtype=torch.long, device=device)
    
    def encode(self, s: str):
        enc = []
        for c in s:
            enc.append(self.encoding[c])
        return enc

    def decode(self, e):
        dec = ''
        for i in e:
            dec += self.decoding[i]
        return dec

    def get_batch(self, train=True,
                batch_size=4,
                context_length=8,):
        data = None
        if train:
            data = self.training
        else:
            data = self.validation
        
        #labels are one offset from the training
        x_idx = np.random.randint(low=0,
                              high=len(data)-context_length-1,
                              size=(batch_size,))
        batch_data = torch.stack([self.training[i:i+context_length] for i in x_idx])
        batch_labels = torch.stack([self.training[i+1:i+context_length+1] for i in x_idx])

        return batch_data, batch_labels
        

    

if __name__ == "__main__":

    d = shakeData('./tinyShake.txt',)
    
