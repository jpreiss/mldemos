import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TOKENS = ["END"] + [str(i) for i in range(10)] + ["+", "="]
TOKIND = {t: i for i, t in enumerate(TOKENS)}
NTOK = len(TOKENS)
DIM = 32
CONTEXT = 5


class MultiAttention(nn.Module):
    def __init__(self, heads, head_dim):
        super().__init__()
        self.heads = heads
        rows = heads * head_dim
        self.Q = nn.Linear(DIM, rows, bias=False)
        self.K = nn.Linear(DIM, rows, bias=False)
        self.V = nn.Linear(DIM, rows, bias=False)
        self.mask = torch.triu(torch.zeros((CONTEXT, CONTEXT)) - np.inf, 1)
        self.proj = nn.Linear(rows, DIM, bias=False)

    def forward(self, X):
        batch, c, d = X.shape
        assert c == CONTEXT
        assert d == DIM
        Qh = self.Q(X)
        Kh = self.K(X)
        Vh = self.V(X)
        Ycat = torch.zeros_like(Vh)
        head_dim = Ycat.shape[-1] // self.heads
        for i in range(self.heads):
            j0 = head_dim * i
            j1 = head_dim * (i + 1)
            Q = Qh[:, :, j0:j1]
            K = Kh[:, :, j0:j1]
            V = Vh[:, :, j0:j1]
            # all are CONTEXT x DIM
            A = Q @ K.transpose(-1, -2) / np.sqrt(head_dim) # attention
            assert A.shape == (batch, CONTEXT, CONTEXT)
            # rows of A correspond to Q's "soft lookup"
            A += self.mask[None, :, :]
            S = torch.softmax(A, dim=-1)
            assert S.shape == A.shape
            Ycat[:, :, j0:j1] = S @ V
        Y = self.proj(Ycat) + X
        return Y


class Transformer(nn.Module):
    def __init__(self, heads, head_dim, layers):
        super().__init__()
        pos_init = torch.normal(mean=0, std=1/np.sqrt(DIM), size=(CONTEXT, DIM))
        self.pos_embed = nn.Parameter(pos_init)
        self.tok_embed = nn.Embedding(NTOK, DIM)
        self.tok_unembed = nn.Linear(DIM, NTOK, bias=False)
        self.layers = [MultiAttention(heads, head_dim) for _ in range(layers)]

    def forward(self, toks):
        batch, c = toks.shape
        assert c == CONTEXT
        X0 = self.tok_embed(toks)
        assert X0.shape == (batch, CONTEXT, DIM)
        X = X0 + self.pos_embed
        for layer in self.layers:
            X = layer(X)
        # now is context x dim
        logits = self.tok_unembed(X)
        # now is context x NTOK, aka logits for dists over tokens
        return logits


def generate(trans, toks, n):
    batch, j = toks.shape
    assert j + n <= CONTEXT
    # zero is END
    x = torch.zeros(batch, CONTEXT, dtype=torch.long)
    x[:, :j] = toks
    out = []
    with torch.no_grad():
        for i in range(n):
            logits = trans(x)[:, j - 1]
            assert logits.shape == (batch, NTOK)
            assert not any(torch.isnan(logits.flatten()))
            dist = F.softmax(logits, dim=-1)
            sample = torch.multinomial(dist, num_samples=1).squeeze()
            assert sample.shape == (batch,)
            out.append(sample)
            x[:, j] = sample
            j += 1
    return torch.stack(out).T


def main():
    # generate some training data
    data_str = []
    for a in range(5):
        for b in range(5):
            c = a + b
            toks = [str(a), "+", str(b), "=", str(c)]
            data_str.append(toks)
    data_int = [np.array([TOKIND[t] for t in toks]) for toks in data_str]
    data_int = torch.LongTensor(np.stack(data_int))
    assert torch.all(data_int[:, 3] == NTOK - 1)

    # TODO: remove final elt
    X_train = data_int
    Y_train = data_int[:, 1:]

    trans = Transformer(heads=1, head_dim=DIM, layers=1)

    epochs = 5000
    opt = torch.optim.AdamW(trans.parameters(), lr=1e-2)
    for epoch in range(epochs):
        opt.zero_grad()
        #logits = trans(d)
        #dists = F.softmax(logits, dim=1)[:-1]
        #onehots = F.one_hot(d[1:], NTOK)
        #print(f"{dists = }\n{onehots = }")
        logits = trans(X_train)[:, :-1]
        if False:
            # DEBUG
            idx = 2
            loss = F.cross_entropy(logits[:, idx], Y_train[:, idx])
        else:
            loss = F.cross_entropy(logits.reshape(-1, NTOK), Y_train.flatten())
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            print(f"After epoch {epoch}, I think...")
            print(f"loss = {loss.item()}")
            with torch.no_grad():
                X = X_train[:, :3]
                Y = generate(trans, X, n=2)
                assert Y.shape[-1] == 2
                XY = torch.cat([X, Y], dim=1)
                errors = torch.sum((XY != X_train).flatten())
                N = X_train.shape[0]
                print(f"errors = {errors}/{N}")
                samples = 10
                idx = np.random.choice(N, size=samples)
                for xy in XY[idx]:
                    print("".join(TOKENS[t] for t in xy))


if __name__ == "__main__":
    main()
