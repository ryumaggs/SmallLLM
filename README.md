# SmallGPT Codebase/Exercise

This repository is based on the walkthrough provided by Andrej Karpathy at [this link](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=6331s)

## Data Set, Requirements, and Training

Trained entirely on the Tiny Shakespeare data set, provided in repo at tinyShake.txt \
\
Only two python libraries are required: Torch and Numpy. \
\
Main.py contains the hyperparemeter settings and main training loop

## Sample Output and Loss

Lowest validation loss achieved: 0.0076 (cross entropy)

Sample Output: \
VOLTARUENCENCK: \
Ocicke jove? \
 \
UKE VINCHANGER: \
Go to king he do lake, sour scempions, his to lefse.

Hath and chathem ose them,\
Come to don the prome,\
To wed gome chord the forsh che ampiest the rith\
That\
The sus'd lep, I beraon; whou is;\
T is the dake the his freincent my thou me he ofn veristate,\
Wh poible this for epa jonk hist fear\
The more timpang, bir with not his,\
:ixhou rojocelr.\
\
DUKE VALT:\
Whout like, bear, whenigh thy ,is the her abs is So with my my of to hoep;\
Ke no, for boved.\


## Model Architecture

This repo implements purely a multi-block transformer decoder.
Importantly: no cross attention implementation

- Transformer.py contains all sub-blocks for building a decoding transformer, with the highest level being a Block()
- Each Block contains:
  - N x Multi-head attention (MHA) units
  - 1 x Feed Forward (FF) MLP
  - Both MHA and FF contain:
    - Dropout, default = 0.2
    - Skip connections
    - Layer norm (pre-norm architecture)

## Hyperparameters
- Data configs
  - batch_size = 64
  - block_size = 256
  - embed_size = 384

- Network architecture configs
  - num_heads = 6
  - num_blocks = 6
  - dropout = 0.2
  - device = 'cuda:0'

- training configs
  - max_iters = 10000
  - learning_rate = 3e-4

