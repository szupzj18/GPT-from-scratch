import torch
import torch.nn as nn
import torch.nn.functional as F
# step1: load the data
with open ("input.txt", "r") as file:
    text=file.read()

# step2: tokenization

chars = tuple(set(text))
int2char=dict(enumerate(chars))
# print("int2char: \n", int2char)
char2int={ch:ii for ii,ch in int2char.items()}
# print("char2int: \n", char2int)

encoder = lambda x: [char2int[ch] for ch in x]
decoder = lambda x: ''.join([int2char[i] for i in x])

print("encoder: \n", encoder("hello"))
print("decoder: \n", decoder(encoder("chris")))

# step3: encoding
data = torch.tensor(encoder(text), dtype=torch.long)
print("data shape: ", data.shape)
print("data type: ", data.dtype)
print("data[1000]: ", data[:1000])
