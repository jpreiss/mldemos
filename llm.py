import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOKENS = ["END"] + [str(i) for i in range(10)] + ["+", "="]
TOKIND = {t: i for i, t in enumerate(TOKENS)}
NTOK = len(TOKENS)
DIM = 32
CONTEXT = 5


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Q = nn.Linear(DIM, DIM, bias=False)
        self.K = nn.Linear(DIM, DIM, bias=False)
        self.V = nn.Linear(DIM, DIM, bias=False)
        self.mask = torch.triu(torch.zeros((CONTEXT, CONTEXT)) - np.inf, 1)

    def forward(self, X):
        Q = self.Q.forward(X)
        K = self.K.forward(X)
        V = self.V.forward(X)
        # all are CONTEXT x DIM
        A = Q @ K.T / np.sqrt(DIM) # attention
        # rows of A correspond to Q's "soft lookup"
        A += self.mask
        S = torch.softmax(A, dim=1)
        assert S.shape == A.shape
        Y = S @ V
        return Y


def pos_encoding(context, dim):
    t = (2 * np.pi / context) * np.tile(np.arange(context), (context, 1)).squeeze()
    assert t.shape == (context, context)
    omegas = np.arange(context)
    fourier = np.cos(t * omegas[:, None])
    assert fourier.shape == (context, context)
    #plt.imshow(fourier)
    #plt.show()
    subsample = np.linspace(0, context - 1, dim).astype(int)
    enc = fourier[:, subsample].astype(np.float32)
    #plt.imshow(enc)
    #plt.show()
    return enc


class Transformer(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # TODO: multi-head
        self.pos_embed = torch.tensor(pos_encoding(CONTEXT, DIM))
        self.tok_embed = nn.Embedding(NTOK, DIM)
        self.layers = [Attention() for _ in range(layers)]
        self.tok_unembed = nn.Linear(DIM, NTOK)

    def forward(self, toks):
        assert toks.shape == (CONTEXT,)
        X0 = self.tok_embed.forward(toks)
        assert X0.shape == (CONTEXT, DIM)
        X = X0 + self.pos_embed
        for layer in self.layers:
            X = layer.forward(X) + X
        # now is context x dim
        logits = self.tok_unembed.forward(X)
        # now is context x NTOK, aka logits for dists over tokens
        return logits


def generate(trans, toks, n):
    j = len(toks)
    assert j + n <= CONTEXT
    # zero is END
    x = torch.LongTensor(np.zeros(CONTEXT))
    x[:j] = toks
    out = []
    with torch.no_grad():
        for i in range(n):
            logits = trans.forward(x)[j]
            assert logits.shape == (NTOK,)
            assert not any(torch.isnan(logits))
            dist = F.softmax(logits).detach().numpy()
            sample = np.random.choice(NTOK, p=dist)
            out.append(sample)
            x[j] = sample
            j += 1
    return np.array(out)


def main():
    # generate some training data
    data_str = []
    for a in range(5):
        for b in range(5):
            c = a + b
            toks = [str(a), "+", str(b), "=", str(c)]
            data_str.append(toks)
    data_int = [np.array([TOKIND[t] for t in toks]) for toks in data_str]
    X_int = torch.LongTensor(np.stack(data_int))
    assert torch.all(X_int[:, 3] == NTOK - 1)
    trans = Transformer(layers=1)

    epochs = 1000
    opt = torch.optim.Adam(trans.parameters(), lr=1e-3)
    for epoch in range(epochs):
        shuf = np.random.permutation(len(X_int))
        X_int = X_int[shuf, :]
        assert torch.all(X_int[:, 3] == NTOK - 1)
        if epoch % 10 == 0:
            print(f"After epoch {epoch}, I think...")
            with torch.no_grad():
                for _ in range(10):
                    idx = np.random.choice(len(data_int))
                    x = X_int[idx] + 0  # copy for inplace edit later
                    y_toks = generate(trans, x[:3], n=2)
                    x[3:] = torch.tensor(y_toks)
                    print("".join(TOKENS[t] for t in x))
        for i in range(len(X_int)):
            d = X_int[i, :]
            #print("".join(data_str[i]))
            opt.zero_grad()
            logits = trans.forward(d)
            dists = F.softmax(logits, dim=1)[:-1]
            onehots = F.one_hot(d[1:], NTOK)
            #print(f"{dists = }\n{onehots = }")
            logits = trans.forward(d)[:-1, :] # can't predict after end
            target = d[1:]
            assert d[3] == NTOK - 1  # equals sign
            # DEBUG target = torch.LongTensor(np.repeat(NTOK - 1, CONTEXT - 1))
            loss = F.cross_entropy(logits, target)
            if False:
                # DEBUG
                loss = F.cross_entropy(logits[2], target[2])
            loss.backward()
            opt.step()


if __name__ == "__main__":
    main()
