import model
import shakeData
import torch
from tqdm import tqdm

def train(model, 
          data, 
          epochs,
          batch_size,
          context_length,):
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss_fn = torch.nn.functional.cross_entropy
    pbar = tqdm(range(epochs))
    for step in pbar:
        x, y = data.get_batch(train=True,
                              batch_size=batch_size,
                              context_length=context_length)
        logits = model(x)
        #logits is (B x T x C)
        # y is (B x T) need to flatten so that its (BT x C) and (BT)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        y = y.view(B*T)
        train_loss = loss_fn(logits,y)
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=f"{train_loss.item():.4f}")

def validate(model,
             data, 
          batch_size,
          context_length,):
    loss_fn = torch.nn.functional.cross_entropy
    with torch.no_grad():
        x, y = data.get_batch(train=False,
                        batch_size=batch_size,
                        context_length=context_length)
        logits = model(x)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        y = y.view(B*T)
        val_loss = loss_fn(logits,y)
        print("val loss: ", val_loss.item())

if __name__ == "__main__":
    #hyper parameter config
    #data configs
    batch_size = 64
    block_size = 256
    embed_size = 384

    #network architecture configs
    num_heads = 6
    num_blocks = 6
    dropout = 0.2
    device = torch.device('cuda:1')

    #training configs
    max_iters = 10000
    learning_rate = 3e-4

    d = shakeData.shakeData('./tinyShake.txt',)
    '''
    model = model.BigramModel(d.vocab_size,
                               block_size=8,
                               head_size=32,
                               num_heads=4,
                               embed_size=32)'''
    model = model.MultiBlockModel(embed_size= embed_size,
                                  num_blocks = num_blocks,
                                num_heads=num_heads,
                                block_size=block_size,
                                vocab_size=d.vocab_size,
                                dropout=0.2,
                                device=torch.device('cuda:1'))

    if False:
        train(model, d, 10000,
              batch_size=batch_size,context_length=block_size)
        torch.save(model.state_dict(), './models/model.pth')
    else:
        model.load_state_dict(torch.load('./models/model.pth'))
        model.eval()
    starting_tokens = [0] * (block_size - 1) + [34]
    output_tokens = model.generate(starting_tokens=starting_tokens,
                   max_new_tokens = 500,
                   context_length=block_size)

    output_decoded = d.decode(output_tokens[block_size - 1:])
    print(output_decoded)

    validate(model, d,
              batch_size=batch_size,context_length=block_size)