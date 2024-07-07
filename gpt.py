import os

print(os.getcwd())

file = "/home/kyusho/gpt/char-rnn/data/tinyshakespeare/input.txt"

with open(file, "r", encoding="utf-8") as f:
    text = f.read()
    
# Length of text
print("Length of dataset in characters: ", len(text))

# First 1000 characters
print("The first 1000 characters:\n", text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("".join(chars)) # .join() needs a separator in form of a string: "some string".join() 
print(vocab_size)

# Mapping from characters to integers, Encoder and Decoder

stoi = { ch:i for i,ch in enumerate(chars) } # Maps character 1 with index 1 etc, enumerate() generates pairs of (index, character)
itos = { i:ch for i,ch in enumerate(chars) } 
encode = lambda s: [stoi[c] for c in s] # Takes string, returns list of indices
decode = lambda l: "".join([itos[i] for i in l]) # Takes list, returns string

print(encode("hii there"))
print(decode(encode("hii there")))

import torch
data = torch.tensor(encode(text))
print(data.shape(), data.dtzpe())