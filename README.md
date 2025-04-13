# GPT From Scratch

YouTube source: https://www.youtube.com/watch?v=kCc8FmEb1nY

## Goal

build a simple language model using pytorch & cuda.

- [x] basic Bigram Model
- [x] v2 with Self-attention
- [x] multi head attention.(in gpt-v2.py)

## Getting started

install deps:
```bash
python3 -m pip install -r requirements.txt
```

run demo:
```bash
python3 gpt-v2.py
```

## Training steps & output overview

```shell
# initial loss.
loss:  tensor(4.1488, grad_fn=<NllLossBackward0>)
# initial generate 100 tokens.
v0 generate:  XRnfD3vrG&LjOVJ:kw!NoQZqld-n3I,S:.uTtI!3dz  KGvky!cAxjNrG'S&mv,DOQGlNmwZdtVIihdaXoRZnkSE
KHyQTBeIvJfK
```

training steps
```shell
step 0: train loss 4.1909, val loss 4.1939
step 1000: train loss 2.5132, val loss 2.5148
step 2000: train loss 2.3986, val loss 2.3927
step 3000: train loss 2.3249, val loss 2.3324
step 4000: train loss 2.2858, val loss 2.3171
step 5000: train loss 2.2624, val loss 2.2963
step 6000: train loss 2.2380, val loss 2.2799
step 7000: train loss 2.2174, val loss 2.2637
step 8000: train loss 2.1955, val loss 2.2494
step 9000: train loss 2.1921, val loss 2.2390
------------
```
after 10000 steps training, loss get to ~2.23.

lets try to get 400 tokens output...

v2 generate: 
```shell
 XES:
Gat?

Pramenawis noch in pho but, spur hand this ser swrand and dithbe rawers dows ale ci' steer?

KING OLERn:
Roorst forsttces? bemes noray.on the hwe us lortt lomine,
Ther, thill ank?

The peeperee rrouelliee frast ught beith this trich let ust me of and Llikesefh: sur in.

Thath and coust
I Hon
Whe danst and ien poparty paid wair
Iflof Coumh of ylo--knot for much saldersent whavelou ap my s

```

still garbage but it looks like model wanna say something :-) haha.
