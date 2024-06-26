"""Applies a decoder-only (GPT-like) Transformer to model math equations."""

import torch
from torch import nn
from torch.nn import functional as F

from util import fastmode


# Tokens are just for integer math equations.
TOKENS = ["END"] + [str(i) for i in range(10)] + ["+", "*", "="]
TOKIND = {t: i for i, t in enumerate(TOKENS)}
NTOK = len(TOKENS)

# Hyperparameters.
DIM = 32
CONTEXT = 6


class MultiAttention(nn.Module):
    """One layer of masked multi-head attention with MLP post-computation."""

    def __init__(self, heads, head_dim, mlp=True):
        super().__init__()
        self.heads = heads
        rows = heads * head_dim
        # All the heads are stored together for efficiency.
        self.Q = nn.Linear(DIM, rows, bias=False)
        self.K = nn.Linear(DIM, rows, bias=False)
        self.V = nn.Linear(DIM, rows, bias=False)
        self.mask = torch.triu(torch.zeros((CONTEXT, CONTEXT)) - float('inf'), 1)
        if mlp:
            self.proj = nn.Sequential(
                nn.LayerNorm(rows),
                nn.Linear(rows, rows * 4, bias=False),
                nn.ReLU(),
                nn.Linear(rows * 4, DIM, bias=False),
            )
        else:
            self.proj = nn.Linear(rows, DIM, bias=False)

    def forward(self, X):
        batch, c, d = X.shape
        assert c == CONTEXT
        assert d == DIM
        def remap(A):
            return A.reshape(batch, c, self.heads, -1).transpose(-2, -3)
        Q = remap(self.Q(X))
        K = remap(self.K(X))
        V = remap(self.V(X))
        head_dim = Q.shape[-1]
        A = Q @ K.transpose(-2, -1) * head_dim**-0.5
        assert A.shape == (batch, self.heads, CONTEXT, CONTEXT)
        A += self.mask[None, None, :, :]
        S = torch.softmax(A, dim=-1)
        assert S.shape == A.shape
        Ycat = S @ V
        Ycat = Ycat.transpose(-3, -2).reshape(batch, CONTEXT, self.heads * head_dim)
        Y = self.proj(Ycat) + X
        return Y


class Transformer(nn.Module):
    def __init__(self, heads, head_dim, layers):
        super().__init__()
        pos_init = torch.normal(mean=0, std=DIM**-0.5, size=(CONTEXT, DIM))
        self.pos_embed = nn.Parameter(pos_init)
        self.tok_embed = nn.Embedding(NTOK, DIM)
        self.tok_unembed = nn.Linear(DIM, NTOK, bias=False)
        self.layers = nn.ModuleList(
            [MultiAttention(heads, head_dim) for _ in range(layers)]
        )
        self.norms = nn.ModuleList(
            [None] + [nn.LayerNorm(DIM) for _ in range(layers - 1)]
        )

    def forward(self, toks):
        batch, c = toks.shape
        assert c == CONTEXT
        X0 = self.tok_embed(toks)
        assert X0.shape == (batch, CONTEXT, DIM)
        X = X0 + self.pos_embed
        for layer, norm in zip(self.layers, self.norms):
            if norm is not None:
                X = norm(X)
            X = layer(X)
        # now is context x dim
        logits = self.tok_unembed(X)
        # now is context x NTOK, aka logits for dists over tokens
        return logits

    def generate(self, toks, n):
        """Samples from the language model autoregressively."""
        batch, j = toks.shape
        assert j + n <= CONTEXT + 1
        # zero is END
        x = torch.zeros(batch, CONTEXT, dtype=torch.long)
        out = []
        with torch.no_grad():
            for i in range(n):
                x[:, :j] = toks
                logits = self(x)[:, j - 1]
                assert logits.shape == (batch, NTOK)
                assert not any(torch.isnan(logits.flatten()))
                dist = F.softmax(logits, dim=-1)
                sample = torch.multinomial(dist, num_samples=1).squeeze()
                assert sample.shape == (batch,)
                out.append(sample)
                toks = torch.cat([toks, sample[:, None]], axis=-1)
                if i != n - 1:
                    x[:, j] = sample
                    j += 1
        return torch.stack(out).T


def print_toks(toks):
    """Converts ints->strings, prints, and stops on END."""
    strs = []
    for t in toks[1:]:
        if t == 0:
            break
        strs += TOKENS[t]
    print("".join(strs))


def check_completion(data_int, trans):
    """Checks performance on completion of the training set starting with ="""
    N = data_int.shape[0]
    samples = 10
    X = data_int[:, :4] # END, a, op, b -- no equals sign
    Y = trans.generate(X, n=3) # should be =, c1, c2
    XY = torch.cat([X, Y], dim=1)
    errors = torch.sum(torch.any(XY != data_int, dim=-1))
    print(f"errors = {errors}/{N}")
    idx = torch.randint(N, size=(samples,))
    for xy in XY[idx]:
        print_toks(xy)


def check_coverage(data_int, trans):
    """Checks performance on generating the training set from nothing."""
    # Expect some duplicates, so generate multiple x of dataset.
    X = torch.zeros(len(data_int) * 5, 1, dtype=torch.long)
    Y = trans.generate(X, n=6)
    assert Y.shape[-1] == 6
    XY = torch.cat([X, Y], dim=1)
    sD = set(tuple(d.tolist()) for d in data_int)
    sXY = set(tuple(xy.tolist()) for xy in XY)

    missed_true = sD - sXY
    n_true = len(sD & sXY)
    print(f"generated {n_true}/{len(sD)} true equations")
    if 0 < len(missed_true) < 20:
        print("missed true equations:")
        for mt in missed_true:
            print_toks(mt)

    false_eqns = sXY - sD
    n_false = len(false_eqns)
    print(f"generated {n_false} false equations")
    if 0 < n_false < 20:
        print("false equations:")
        for f in false_eqns:
            print_toks(f)

    print("gen samples:")
    samples = 10
    idx = torch.randint(data_int.shape[0], size=(samples,))
    for xy in XY[idx]:
        print_toks(xy)


def main():
    # Generate training data of equations.
    data_str = []
    for a in range(10):
        for b in range(10):
            for c, op in zip([a + b, a * b], ["+", "*"]):
                toks = ["END", str(a), op, str(b), "="]
                if c > 9:
                    toks += [str(c // 10), str(c % 10)]
                else:
                    toks += [str(c), "END"]
                data_str.append(toks)

    # Convert to tensors.
    data_int = [[TOKIND[t] for t in toks] for toks in data_str]
    data_int = torch.LongTensor(data_int)
    assert torch.all(data_int[:, 4] == NTOK - 1)
    X_train = data_int[:, :-1]
    Y_train = data_int[:, 1:]

    # Construct model and print some info.
    trans = Transformer(heads=4, head_dim=8, layers=1)
    params = sum(p.numel() for p in trans.parameters())
    datapoints = X_train.shape[0]
    print(
        f"{params = }, {datapoints = }"
        f" (overparameterized by {params / datapoints:.2f}x)"
    )

    # Train model.
    epochs = 1001 if fastmode() else 4001
    opt = torch.optim.AdamW(trans.parameters(), lr=1e-3)
    for epoch in range(epochs):
        opt.zero_grad()
        logits = trans(X_train)
        loss = F.cross_entropy(logits.reshape(-1, NTOK), Y_train.flatten())
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            print(f"After epoch {epoch}, I think...")
            print(f"loss = {loss.item()}")
            with torch.no_grad():
                check_completion(data_int, trans)
        if epoch % 1000 == 0:
            print(f"Coverage test for epoch {epoch}:")
            with torch.no_grad():
                check_coverage(data_int, trans)


if __name__ == "__main__":
    main()
