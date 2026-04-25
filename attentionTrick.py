import torch

if __name__ == "__main__":
    B = 4
    T = 8
    C = 32

    x = torch.randn(B,T,C)

    #for loop basis, we want to create a BOW for each time step t IN T
    #such that the channels are averaged over time up until T
    #for each batch
    x_ineff_bow = torch.zeros((B,T,C))
    for b in range(B):
        for t in range(T):
            up_to = x[b,:t+1,:]
            x_ineff_bow[b,t] = torch.mean(up_to,0)
    

    #matrix multiplication basis
    #we want a matrix that is T x T lower triangle where each row sums up to

    #this is a single head. 
    #paramterized by two torch linear layers
    #key and query, which map from channels (which is vocab size?) to
    # the head size, 16

    '''
    so this is very interesting. on a char level lets say im
    only using the 26 letters of the alphabet + space, so 27 chars
    total in my whole vocabulary

    i have a context length of 8, so i look at 8 chars at a time 

    B = 4, T = 8, vocab_size = 27, embed_size=32

    the beginning steps are a bit confusing. 
    first step is i take each token i am working with (in this case char)
    and i embed it using my token table: vocab_size x embed_size

    this produces a B x T x 32 matrix where for each time step i have an 
    embedding for each token/char

    the attention head then allows for tokens to "communicate" with one another
    by the following:
    ----------------------------
    head_size = 16
    key = embed_size x head_size
    query = embed_size x head_size
    value = embed_size x head_size

    operations:
    key(batch) -> B x T x head_size
    query(batch) -> B x T x head_size
    value(batch) -> B x T x head_size

    affinities: query @ key.transpose(-2,-1) -> B x T x T
        each element here says. for batch b
        and a given time step t. how well does it affine
        with other time step tokens
    
        affinities is then masked lower triangle wise
        to make sure that tokens are only talking to past tokens
        and not future tokens
    
        affinities softmaxed for probability, to show which tokens have highest interest
        to a given token. 
    
        output: wei (masked affinities) @ v
            where v = value(batch)

            this ensures that we stay within the head_size
            and is just an abstraction or feature extraction
            using a seperate value layer of the data 
        
            final output shape: B x T x head_size

    -------------------------
    '''
    head_size = 16
    key = torch.nn.Linear(C, head_size, bias=False)
    query = torch.nn.Linear(C, head_size, bias=False)
    value = torch.nn.Linear(C, head_size, bias=False)
    k = key(x) # B x T x head_size
    q = query(x) # B x T x head_size

    print(q.shape)
    print(k.transpose(-2,-1).shape)
    wei = q @ k.transpose(-2,-1) #B x T x T

    print("AA: ", wei.shape)

    mask = torch.tril(torch.ones((T,T)))
    wei = wei.masked_fill(mask==0,float('-inf'))
    wei = torch.nn.functional.softmax(wei,dim=-1)

    v = value(x)
    out = wei @ v
    print(out.shape)
    #----------------------------


