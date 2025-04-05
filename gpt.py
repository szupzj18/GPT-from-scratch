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


block_size = 8 # length of sequence
batch_size = 4 # batch size
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

# batch function to create training and validation batches
# return: context batch and validation target values
def get_batch(split):
    data = train_data if split == 'train' else val_data
    # random sample batch_size random indices
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
        print("idx shape: ", idx.shape)
        
        # logits: the prediction for the next character
        # idx: the input character (B, T) (batch_size, block_size)
        # target: the target character (B, T) (batch_size, block_size)
        # logits: (batch_size, block_size, vocab_size)
        logits = self.token_embedding_table(idx) # B, T, C 4, 8, 65
            
        if target is None:
            loss = None
        else:
            print("target shape: ", target.shape)
            
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
        for _ in range(max_new_tokens):
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
logits, loss = model(xb, yb)
print("logits shape: ", logits.shape)
print("loss: ", loss)
# 使用 [[0]] 来初始化模型的输入
# 这里的 [[0]] 是一个 batch size 为 1 的输入，用来初始化模型输入，然后连续生成 100 个字符
# 最后 tolist 将张量转换为列表
output = model.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()
print("v0 generate: ", decoder(output))

# step6: training
