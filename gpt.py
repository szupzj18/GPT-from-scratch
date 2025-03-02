import torch
import torch.nn as nn
import torch.nn.functional as F
# use GPU if available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# step1: load the data
with open ("input.txt", "r") as file:
    text=file.read()

# step2: tokenization

chars = tuple(set(text))
vocab_size = len(chars)
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
# print("data[1000]: ", data[:1000])

n = int(0.9*len(data)) # 90% of the data for training
train_data, val_data = data[:n], data[n:] # last 10% for validation


block_size = 8 # length of sequence
batch_size = 256 # batch size
# step4: create batches
print("first block: ", train_data[:block_size+1])

# demo code to show how to create a batch
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    # t for time dimension
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context}, target is {target}")

# batch function to create training and validation batches
# return: context batch and validation target values
def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,)) # random idx for batch split
    context = torch.stack([data[i:i+block_size] for i in idx]) 
    target = torch.stack([data[i+1:i+block_size+1] for i in idx]) # target is the next character
    return context, target

xb, yb = get_batch('train')
print("context shape: ", xb.shape)
print("context :", xb)
print("target shape: ", yb.shape)
print("target :", yb)

# decode
# print("context decoed: ", decoder(xb[0].tolist()))
print("validation ")
for batch in range(batch_size):
    for t in range(block_size):
        print(f"when input is {xb[batch][:t+1]}, target is {yb[batch][t]}")

# step5: model
torch.manual_seed(42)
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, target):
        logits = self.token_embedding_table(idx)
        loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_new_tokens=100):
        for _ in range(max_new_tokens):
            

model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)
print("loss: ", loss)

output = model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()
print("v0 generate: ", decoder(output))