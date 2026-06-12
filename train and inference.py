
import torch
import torch.nn as nn
import io

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)

filepath='/kaggle/input/datasets/aboubacardiarra1810/text-for-my-first-lstm/text.txt'
with io.open(filepath ,'r') as f:
    text=f.read()

print(text[:300])

text=text.replace('é','e')

p=int(0.90*len(text))
train_text=text[:p]
val_text=text[p:]

chars=sorted(list(set(text)))
unique_letters=''.join(chars)
print(f'unique letters :{unique_letters}')
len(unique_letters)

char_to_ix = { char: i for i, char in enumerate(chars) }
ix_to_char = { i: char for i, char in enumerate(chars) }
def encode(s):
    return [char_to_ix[c] for c in s ]
def decode(s):
    return ''.join([ix_to_char[d] for d in s])

print(encode("bonjour"))

print(decode([52, 65, 64, 60, 65, 71, 68]
))

#encode the whole text
encoded=torch.tensor(encode(train_text),dtype=torch.long)

X=[]
Y=[]
block_size=8
for i in range (len(encoded)-block_size):
    X.append(encoded[i:i+block_size])
    Y.append(encoded[i+1:1+block_size+i])
X=torch.stack(X)
Y=torch.stack(Y)

class LSTM(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size,vocab_size)
    def forward(self,x):
        x=self.embedding(x)
        out,_=self.lstm(x)
        out=self.fc(out)
        return out

embed_size=64
hidden_size=128
vocab_size=len(chars)

vocab_size

model=LSTM(vocab_size,64,128).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

epochs=10
for epoch in range(epochs):
    running_loss=0
    model.train()
    for x,y in zip(X,Y):
        x,y=x.to(device),y.to(device)
        output = model(x.unsqueeze(0))
        loss=criterion(output.view(-1,vocab_size),y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    if epoch%2==0:
        print(f'epoch:{epoch+1} loss:{running_loss/len(X)}')
        

val_text[:1]

def generating(model,prompt,lenght=200):
    model.eval()
    device="cuda" if torch.cuda.is_available() else "cuda"
    prompt=prompt
    # encoded=torch.tensor(encode(prompt),dtype=torch.long).to(device)
    generated_tokens = encode(prompt)
    with torch.inference_mode():
        for _ in range(lenght):
            context = generated_tokens[-block_size:]
            encoded = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            logits=model(encoded)
            next_logits=logits[0,-1,:]
            proba=torch.softmax(next_logits,dim=-1)
            next_idx=torch.multinomial(proba,num_samples=1).item()
            generated_tokens.append(next_idx)
    return decode(generated_tokens)

print(generating(model,val_text[:1], len(val_text)))