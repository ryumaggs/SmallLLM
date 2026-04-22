import bigram
import shakeData
import torch
from tqdm import tqdm


def train(model, data, epochs):
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss_fn = torch.nn.functional.cross_entropy
    pbar = tqdm(range(epochs))
    for step in pbar:
        x, y = data.get_batch(train=True,
                              batch_size=32,)
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


if __name__ == "__main__":
    d = shakeData.shakeData('./tinyShake.txt',)
    model = bigram.BigramModel(d.vocab_size)

    train(model, d, 10000)
    output_tokens = model.generate(starting_tokens = [34],
                   max_new_tokens = 100,)
    
    output_decoded = d.decode(output_tokens)
    print(output_tokens)
    print(output_decoded)