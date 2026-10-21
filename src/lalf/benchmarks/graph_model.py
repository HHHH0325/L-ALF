"""Small molecular GCN baseline with native PyTorch."""

import copy
import numpy as np
import torch
from rdkit import Chem
from .core import COLS, transform, untransform

torch.set_num_threads(2)


def graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    features = []
    edges = []
    elements = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    for a in mol.GetAtoms():
        features.append(
            [float(a.GetAtomicNum() == z) for z in elements]
            + [
                a.GetAtomicNum() / 100,
                a.GetDegree() / 6,
                a.GetFormalCharge() / 3,
                float(a.GetIsAromatic()),
                a.GetTotalNumHs() / 4,
            ]
        )
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        edges.extend([(i, j), (j, i)])
    edges.extend((i, i) for i in range(len(features)))
    return np.array(features, np.float32), np.array(edges, np.int64).T


def pack(graphs, device):
    xs = []
    es = []
    bs = []
    offset = 0
    for i, (x, e) in enumerate(graphs):
        xs.append(x)
        es.append(e + offset)
        bs.extend([i] * len(x))
        offset += len(x)
    return (
        torch.tensor(np.concatenate(xs), device=device),
        torch.tensor(np.concatenate(es, axis=1), device=device),
        torch.tensor(bs, device=device),
        len(graphs),
    )


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(14, 48)
        self.conv = torch.nn.ModuleList([torch.nn.Linear(48, 48) for _ in range(3)])
        self.output = torch.nn.Sequential(
            torch.nn.Linear(48, 48), torch.nn.ReLU(), torch.nn.Linear(48, 4)
        )

    def forward(self, packed):
        x, e, b, n = packed
        h = torch.relu(self.input(x))
        src, dst = e
        degree = torch.bincount(dst, minlength=len(x)).clamp_min(1).to(h.dtype)
        norm = (degree[src] * degree[dst]).rsqrt()[:, None]
        for layer in self.conv:
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, h[src] * norm)
            h = torch.relu(layer(agg))
        pooled = torch.zeros(n, 48, device=x.device)
        pooled.index_add_(0, b, h)
        return self.output(pooled / torch.bincount(b, minlength=n).clamp_min(1)[:, None])


def graph_predict(lib, history, ids, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = [graph(s) for s in history.SMILES]
    test_graphs = [graph(lib.smiles[int(i)]) for i in ids]
    y = transform(history[COLS].to_numpy(float))
    center = y.mean(0)
    scale = np.maximum(y.std(0), 1e-6)
    yn = (y - center) / scale
    predictions = []
    epochs = []
    for member in range(3):
        torch.manual_seed(seed * 101 + member)
        rng = np.random.default_rng(seed * 101 + member)
        perm = rng.permutation(len(y))
        nv = min(len(y) - 1, max(1, len(y) // 5))
        val = perm[:nv]
        tr = perm[nv:]
        boot = rng.choice(tr, len(tr), replace=True)
        model = Net().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        vp = pack([train[i] for i in val], device)
        vy = torch.tensor(yn[val], dtype=torch.float32, device=device)
        best = float("inf")
        stale = 0
        state = None
        for epoch in range(80):
            model.train()
            rng.shuffle(boot)
            for start in range(0, len(boot), 64):
                batch = boot[start : start + 64]
                xp = pack([train[i] for i in batch], device)
                target = torch.tensor(yn[batch], dtype=torch.float32, device=device)
                opt.zero_grad()
                loss = (model(xp) - target).square().mean()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float((model(vp) - vy).square().mean().item())
            if vl < best - 1e-4:
                best = vl
                state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 12:
                break
        model.load_state_dict(state)
        model.eval()
        epochs.append(epoch + 1)
        with torch.no_grad():
            pred = np.concatenate(
                [
                    model(pack(test_graphs[start : start + 512], device)).cpu().numpy()
                    for start in range(0, len(ids), 512)
                ]
            )
            predictions.append(pred * scale + center)
    return untransform(np.array(predictions)), {
        "kind": "graph",
        "architecture": "3-layer normalized GCN, mean pooling, atom features, no bond-type features",
        "ensemble_members": 3,
        "epochs": epochs,
        "uncertainty": "bootstrap ensemble dispersion",
        "validation": "20% of queried history; early stopping",
    }
