#import tensor from PyTorch library
import torch
import torch.nn as nn
from torch.nn import functional as F

#parameters
batch_size = 64
block_size = 256
eval_iters = 200
learning_rate = 1e-3
max_iters = 5000
eval_interval = 500
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2 #20% of all intermediate calculations are dropped to 0

with open('article1.txt', 'r', encoding='utf-8') as readfile:
    text = readfile.read()
#list of all characters in dataset sorted
characters = sorted(list(set(text)))
character_size = len(characters) #options of all characters model can emit

#tokenization, other options for tokenizers include sentencepiece (Google), tiktoken
stoi = {ch: i for i, ch in enumerate(characters)}
itos = {i:ch for i, ch in enumerate(characters)}
encode = lambda st: [stoi[c] for c in st] #converts string(input to lambda function) to list of integers
decode = lambda listint: ''.join([itos[i] for i in listint]) #converts list of integers (input to lambda function) to string

data = torch.tensor(encode(text), dtype = torch.long)

#measure how much our model overfits - allocate 90% data to training 
n = int(0.9*len(data))
training_data = data[:n]
val_data = data[n:]

#work with chunks of data a time, train random chunks at a time
# max length = context_length

#not looked at
context_length = 8
training_data[:context_length]

x = training_data[:context_length] #input to transformer
y = training_data[1:context_length+1]
for t in range(context_length):
    context = x[:t+1]
    target = y[t]

#sampling random locations from dataset to retrieve chunks
torch.manual_seed(1457)
batch_size = 4 #independent sequences that will be processed simultaneously
context_length = 8 #max context length
# not looked at

def get_batch(split):
    if split =='train':
        data = training_data 
    else:
        data = val_data
    ix = torch.randint(len(data) - context_length, (context_length,)) #generate random positions to retrieve chunks
    x = torch.stack([data[i:i+context_length] for i in ix]) #first context_length characters starting at i
    y = torch.stack([data[i+1:i+context_length+1] for i in ix]) #stack 1D tensors into row (4X8 tensor)
    return x, y

@torch.no_grad() #don't call .backward() in this function - more efficient memory use since past parameters do not need to be used
def estimate_loss():
    out = {}
    m.eval() #evaluation phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = m(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train() #training phase
    return out

torch.manual_seed(1457) #reproducability

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #creating lower triangular matrix
        self.dropout = nn.Dropout(dropout)    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C **-0.5 #normalize and use scaled attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #communicate with the past
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) #aggregate value and output
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        #return torch.cat([h(x) for h in self.heads], dim=-1) #transformation of outcome of layer
        return out
   
class FeedForward(nn.Module):
    #single layer
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4*n_embed), nn.ReLU(), nn.Linear(4*n_embed, n_embed), nn.Dropout(dropout)) #second-to-;ast last parameter is going back into the residual pathway
    def forward(self, x): #on a per token level, think on data from self attention individually
        return self.net(x)
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x+ self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) #computation done on each self token individually, fork into side computation and return to pipeline --> residual connections            |          credit: Andrej Karpathy
import torch
import torch.nn as nn
from torch.nn import functional as F

#parameters
batch_size = 64
block_size = 256
eval_iters = 200
learning_rate = 1e-3
max_iters = 5000
eval_interval = 500
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2 #20% of all intermediate calculations are dropped to 0

with open('article1.txt', 'r', encoding='utf-8') as readfile:
    text = readfile.read()
#list of all characters in dataset sorted
characters = sorted(list(set(text)))
character_size = len(characters) #options of all characters model can emit

#tokenization, other options for tokenizers include sentencepiece (Google), tiktoken
stoi = {ch: i for i, ch in enumerate(characters)}
itos = {i:ch for i, ch in enumerate(characters)}
encode = lambda st: [stoi[c] for c in st] #converts string(input to lambda function) to list of integers
decode = lambda listint: ''.join([itos[i] for i in listint]) #converts list of integers (input to lambda function) to string

data = torch.tensor(encode(text), dtype = torch.long)

#measure how much our model overfits - allocate 90% data to training 
n = int(0.9*len(data))
training_data = data[:n]
val_data = data[n:]

#work with chunks of data a time, train random chunks at a time
# max length = context_length

#not looked at
context_length = 8
training_data[:context_length]

x = training_data[:context_length] #input to transformer
y = training_data[1:context_length+1]
for t in range(context_length):
    context = x[:t+1]
    target = y[t]

#sampling random locations from dataset to retrieve chunks
torch.manual_seed(1457)
batch_size = 4 #independent sequences that will be processed simultaneously
context_length = 8 #max context length
# not looked at

def get_batch(split):
    if split =='train':
        data = training_data 
    else:
        data = val_data
    ix = torch.randint(len(data) - context_length, (context_length,)) #generate random positions to retrieve chunks
    x = torch.stack([data[i:i+context_length] for i in ix]) #first context_length characters starting at i
    y = torch.stack([data[i+1:i+context_length+1] for i in ix]) #stack 1D tensors into row (4X8 tensor)
    return x, y

@torch.no_grad() #don't call .backward() in this function - more efficient memory use since past parameters do not need to be used
def estimate_loss():
    out = {}
    m.eval() #evaluation phase
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = m(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train() #training phase
    return out

torch.manual_seed(1457) #reproducability

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #creating lower triangular matrix
        self.dropout = nn.Dropout(dropout)    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C **-0.5 #normalize and use scaled attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #communicate with the past
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) #aggregate value and output
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        #return torch.cat([h(x) for h in self.heads], dim=-1) #transformation of outcome of layer
        return out
   
class FeedForward(nn.Module):
    #single layer
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embed, 4*n_embed), nn.ReLU(), nn.Linear(4*n_embed, n_embed), nn.Dropout(dropout)) #second-to-;ast last parameter is going back into the residual pathway
    def forward(self, x): #on a per token level, think on data from self attention individually
        return self.net(x)
class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        x = x+ self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) #computation done on each self token individually, fork into side computation and return to pipeline --> residual connections
        return x
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(character_size, n_embed) #wrapper that uses integers to pluck out corresponding row
        #we need to convert the  token-emb to logits
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        #self.lm_head = nn.Linear(n_embed, character_size)
        #self.sa_heads = MultiHeadAttention(4, n_embed//4) #group convolution
        #self.sa_head = Head(n_embed)
        self.blocks = nn.Sequential(Block(n_embed, n_head=4), Block(n_embed, n_head=4), Block(n_embed, n_head=4), nn.LayerNorm(n_embed))
        #self.ffwd = FeedForward(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, character_size)
    def forward(self, idx, targets = None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) #score for next character in sequence, predicting what character comes next based on identity of given token
        pos_emb = self.position_embedding_table(torch.arange(T, device=None)) #integers from 0--> T-1 are embedded in the table to create TXC
        x = token_emb + pos_emb #holds token identity and position
        #x = self.sa_head(x) #feed into self attention head
        x = self.blocks(x)
        x = self.ln_f(x)
        #x = self.ffwd(x)
        logits = self.lm_head(x) #language model head is fed the output
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #how well is the next character being predicted
        return logits, loss
    def generate(self, idx, max_new_tokens):
        #take B*T to extend to B*T+1, B*T+2, .. max_new_tokens
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #crop context to never pass block_size num of elements
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:] #pluck last element in time decision with -1
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel()
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
context = torch.zeros((1,1), dtype=torch.long, device=None)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


#courtesy, credit: Andrej Karpathy
        return x
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(character_size, n_embed) #wrapper that uses integers to pluck out corresponding row
        #we need to convert the  token-emb to logits
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        #self.lm_head = nn.Linear(n_embed, character_size)
        #self.sa_heads = MultiHeadAttention(4, n_embed//4) #group convolution
        #self.sa_head = Head(n_embed)
        self.blocks = nn.Sequential(Block(n_embed, n_head=4), Block(n_embed, n_head=4), Block(n_embed, n_head=4), nn.LayerNorm(n_embed))
        #self.ffwd = FeedForward(n_embed)
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, character_size)
    def forward(self, idx, targets = None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) #score for next character in sequence, predicting what character comes next based on identity of given token
        pos_emb = self.position_embedding_table(torch.arange(T, device=None)) #integers from 0--> T-1 are embedded in the table to create TXC
        x = token_emb + pos_emb #holds token identity and position
        #x = self.sa_head(x) #feed into self attention head
        x = self.blocks(x)
        x = self.ln_f(x)
        #x = self.ffwd(x)
        logits = self.lm_head(x) #language model head is fed the output
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #how well is the next character being predicted
        return logits, loss
    def generate(self, idx, max_new_tokens):
        #take B*T to extend to B*T+1, B*T+2, .. max_new_tokens
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] #crop context to never pass block_size num of elements
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:] #pluck last element in time decision with -1
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel()
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
context = torch.zeros((1,1), dtype=torch.long, device=None)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


#courtesy, credit: Andrej Karpathy

