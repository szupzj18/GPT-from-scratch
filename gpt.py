import torch
import torch.nn as nn
import torch.nn.functional as F
# hyperparameters
block_size = 8 # length of sequence
batch_size = 4 # batch size
learning_rate = 1e-3 # learning rate
max_iters = 10000 # number of iterations
eval_interval = 1000 # evaluation interval
eval_iters = 200 # number of evaluation iterations
# use GPU if available.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)
# NOTE: this is a simple bigram model, which is a simple linear model
# that predicts the next character based on the previous character

# this tutorial is based on previous "make more" series
# https://www.youtube.com/watch?v=8rj0g2c4v6E&list=PLoROMvodv4rO2Xk1a3b7d5c9e8f3a1a5b&index=1

# step1: load the data
with open ("input.txt", "r") as file:
    text=file.read()

# step2: tokenization

chars = tuple(set(text))
vocab_size = len(chars)
print("vocab size: ", vocab_size)
int2char=dict(enumerate(chars))
# print("int2char: \n", int2char)
char2int={ch:ii for ii,ch in int2char.items()}
# print("char2int: \n", char2int)

encoder = lambda x: [char2int[ch] for ch in x]
decoder = lambda x: ''.join([int2char[i] for i in x])

# print("encoder: \n", encoder("hello"))
# print("decoder: \n", decoder(encoder("chris")))

# step3: encoding
data = torch.tensor(encoder(text), dtype=torch.long)
# print("data shape: ", data.shape)
# print("data type: ", data.dtype)
# print("data[1000]: ", data[:1000])

n = int(0.9*len(data)) # 90% of the data for training
train_data, val_data = data[:n], data[n:] # last 10% for validation

# step4: create batches
print("first block: ", train_data[:block_size+1])

# demo code to show how to create a batch
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    # t for time dimension
    context = x[:t+1]
    target = y[t]
    # print(f"when input is {context}, target is {target}")

# data loader
# batch function to create training and validation batches
# return: context batch and validation target values
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # random sample batch_size random indices
    idx = torch.randint(len(data) - block_size, (batch_size,)) # random idx for batch split
    context = torch.stack([data[i:i+block_size] for i in idx]) 
    target = torch.stack([data[i+1:i+block_size+1] for i in idx]) # target is the next character
    context = context.to(device) # move to GPU if available
    target = target.to(device) # move to GPU if available
    return context, target

xb, yb = get_batch('train')
print("context shape: ", xb.shape)
print("context :", xb)
print("target shape: ", yb.shape)
print("target :", yb)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # set the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # set the model back to training mode
    return out
    
# decode
# print("context decoed: ", decoder(xb[0].tolist()))
# print("validation ")
# for batch in range(batch_size):
#     for t in range(block_size):
#         print(f"when input is {xb[batch][:t+1]}, target is {yb[batch][t]}")

# step5: model
torch.manual_seed(42)
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # 在 bigrammodel 中 vocab_size 和 embedding_dim 一样，是因为我们本质上是通过前一个字符来预测下一个字符
        # 就是一个简单的线性模型/分类器
        # Example:
        # 1. One-hot encoding: Embedding(vocab_size, vocab_size)
        # 2. Bigram model: Embedding(vocab_size, vocab_size)
        # 3. GPT model: Embedding(vocab_size, embedding_dim) + Linear(embedding_dim, vocab_size)
        # about linear, I made a small lab to show how linear works: https://colab.research.google.com/drive/1uU_lbhfzE7LTVzTdr8fuiu4JspGAOY6e#scrollTo=O6MVsRwbxMXE
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, target=None):
        # print("idx shape: ", idx.shape)
        
        # logits: the prediction for the next character
        # idx: the input character (B, T) (batch_size, block_size)
        # target: the target character (B, T) (batch_size, block_size)
        # logits: (batch_size, block_size, vocab_size)
        logits = self.token_embedding_table(idx) # B, T, C 4, 8, 65
            
        if target is None:
            loss = None
        else:
            # print("target shape: ", target.shape)
            
            # F.cross_entropy requires (B, T, C) to be reshaped to (B*T, C)
            # and target to be reshaped to (B*T,)
            # B, T, C = logits.shape
            # logits = logits.view(B*T, C) # B*T, C
            # target = target.view(B*T) # B*T,
            # NOTE: target.view(-1) is same as target.view(B*T)
            # print("logits.view(-1) shape: ", logits.view(-1).shape);
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1)) # B*T, C
        return logits, loss
    
    def generate(self, idx, max_new_tokens=100):
        # 作为一个 bigram model，这里其实可以优化为每次只从上一个字符中预测下一个字符，不需要输入整个序列
        # 相当于历史信息没有被使用，但是未来可以扩展到类似 GPT 的模型
        for _ in range(max_new_tokens):
            # idx: (B, T)
            # logits: (B, T, C)
            logits, loss = self(idx, None)
            logits = logits[:, -1, :] # (B, C) only use the last time step to predict the next character
            # NOTE: logits 最终 softmax 之前，可以通过 temperature 参数来控制生成内容的采样分布。
            # temperature = 1 表示原始的分布，temperature < 1 表示更集中（更保守），temperature > 1 表示更平滑（更发散）
            # e.g. temperature = 0.8
            # logits = logits / 0.8 # temperature scaling
            probs = F.softmax(logits, dim=-1) # generate the probability distribution
            # sample from the distribution, which means it's not deterministic, its random but based on the probability
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # contact predict to origin idx (B, T+1)
            
        return idx

model = BigramLanguageModel(vocab_size)
model = model.to(device) # move to GPU if available
logits, loss = model(xb, yb)
print("logits shape: ", logits.shape)
print("loss: ", loss)
# 使用 [[0]] 来初始化模型的输入
# 这里的 [[0]] 是一个 batch size 为 1 的输入，用来初始化模型输入，然后连续生成 100 个字符
# 最后 tolist 将张量转换为列表
output = model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()
print("v0 generate: ", decoder(output))

# step6: training

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

batch_size = 32

for steps in range(max_iters):
    # get batch
    xb, yb = get_batch('train')
    # forward pass
    logits, loss = model(xb, yb)
    # backward pass
    optimizer.zero_grad(set_to_none=True) # set_to_none=True is a new feature in pytorch 1.12
    loss.backward()
    optimizer.step()
    
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
# step 7: output the model
context = torch.zeros((1, 1), dtype=torch.long, device=device) # (B, T)
output = model.generate(idx=context, max_new_tokens=400)[0].tolist()
print("------------\n")
print("v2 generate: \n", decoder(output))
