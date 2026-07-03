"""
gnnexplainer.py
=========================
Companion script for the seminar report:
  "GNNExplainer: Generating Explanations for Graph Neural Networks"

Experiment: Multi-Size Clique Detection + Color Anchor
  Cliques of sizes k=3, 4, 5 are embedded in a sparse Erdos-Renyi base graph
  (binary node classification). Because any sub-clique of a k-clique is itself
  a locally consistent explanation, this dataset directly exposes the local-
  optima failure mode of gradient-based mask optimization: the optimizer
  converges to a single edge (2-clique) local minimum instead of the full
  k-clique, and the problem worsens as k grows.

  Every node also carries two constant "color" features, color1 = 1 ("blue")
  and color2 = 2 ("red"), with the exact same value on every single node.
  Both statements ("color1 = 1", "color2 = 2") are therefore true for every
  node regardless of its class, so neither carries any information about the
  class. These are "anchor" features (see the report's Related Work
  section): a self-explainable GNN could, in principle, report color1 as the
  explanation for clique nodes and color2 as the explanation for non-clique
  nodes, and reach perfect accuracy by secretly deciding which color to
  report using the real (hidden) clique structure, even though color1 and
  color2 themselves are completely non-discriminative. We test whether
  GNNExplainer's feature mask ever falls for a milder version of this same
  trick.

Requirements
------------
    pip install torch torch_geometric networkx matplotlib numpy

Tested with torch==2.12, torch_geometric==2.8, Python==3.13.
"""
import warnings; warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
import networkx as nx
import numpy as np

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


# ── 1. Dataset ─────────────────────────────────────────────────────────────

def build_multi_clique_dataset(n_nodes=200, er_p=0.02, seed=SEED,
                               clique_configs=None):
    """
    Cliques of sizes 3, 4, 5 embedded in a sparse ER graph.
    Binary labels (0=non-clique, 1=clique node).
    Each node also gets two constant color features, color1=1 and color2=2,
    with the same value on every node. Since neither feature ever varies,
    neither can carry any information about clique membership -- our
    stand-in for an "anchor" feature.
    Returns per-node ground-truth edge sets and clique-size mapping.
    """
    if clique_configs is None:
        clique_configs = [(3, 8), (4, 6), (5, 5)]

    rng  = np.random.default_rng(seed)
    G    = nx.erdos_renyi_graph(n_nodes, er_p, seed=seed)
    labels = np.zeros(n_nodes, dtype=int)
    clique_groups = []
    node_gt, node_clique_size = {}, {}
    used = set()

    for csize, n_cliques in clique_configs:
        for _ in range(n_cliques):
            available = [v for v in range(n_nodes) if v not in used]
            if len(available) < csize:
                break
            nodes = list(rng.choice(available, csize, replace=False).astype(int))
            clique_gt = set()
            for i in range(csize):
                for j in range(i+1, csize):
                    u, v = nodes[i], nodes[j]
                    G.add_edge(u, v)
                    clique_gt.update([(u,v),(v,u)])
            for node in nodes:
                labels[node] = 1
                used.add(node)
                node_gt[node]          = clique_gt
                node_clique_size[node] = csize
            clique_groups.append((csize, nodes))

    edges = list(G.edges())
    src = [u for u,v in edges]+[v for u,v in edges]
    dst = [v for u,v in edges]+[u for u,v in edges]
    ei  = torch.tensor([src, dst], dtype=torch.long)

    # Two constant color features, identical on every node: color1 is
    # always 1 ("blue"), color2 is always 2 ("red"). Neither one ever
    # changes value, so neither can tell clique nodes from non-clique nodes.
    x = torch.zeros(n_nodes, 2, dtype=torch.float)
    x[:, 0] = 1.0
    x[:, 1] = 2.0

    y   = torch.tensor(labels, dtype=torch.long)
    return (Data(x=x, edge_index=ei, y=y, num_nodes=n_nodes),
            clique_groups, node_gt, node_clique_size)


# ── 2. Model ────────────────────────────────────────────────────────────────

class GCN(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch, n_layers=3, dropout=0.5):
        super().__init__()
        self.drop = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_ch, hid_ch))
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(hid_ch, hid_ch))
        self.convs.append(GCNConv(hid_ch, out_ch))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.drop, training=self.training)
        return F.log_softmax(self.convs[-1](x, edge_index), dim=-1)


# ── 3. Training ─────────────────────────────────────────────────────────────

def train_model(model, data, n_epochs=300, lr=1e-2, wd=5e-4,
                class_weights=None, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for ep in range(1, n_epochs+1):
        model.train(); opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y, weight=class_weights)
        loss.backward(); opt.step()
        if verbose and ep % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(1)
                acc  = (pred == data.y).float().mean().item()
            print(f"    ep {ep:4d} | loss {loss.item():.4f} | acc {acc:.4f}")
            model.train()
    return model


# ── 4. Evaluation ───────────────────────────────────────────────────────────

def edge_f1(edge_mask, edge_index, gt_edges, threshold=0.5):
    mn  = edge_mask.detach().cpu().numpy()
    ei  = edge_index.cpu().numpy()
    pp  = {(int(ei[0,i]), int(ei[1,i])) for i in range(ei.shape[1])
           if mn[i] > threshold}
    tp  = len(pp & gt_edges)
    fp  = len(pp - gt_edges)
    fn  = len(gt_edges - pp)
    p   = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    r   = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1  = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return p, r, f1


def expl_clique_size(edge_mask, edge_index, true_nodes, threshold=0.5):
    """
    Largest clique found among true_nodes in the thresholded explanation.
    """
    mn   = edge_mask.detach().cpu().numpy()
    ei   = edge_index.cpu().numpy()
    cset = set(true_nodes)
    G    = nx.Graph(); G.add_nodes_from(true_nodes)
    for i in range(ei.shape[1]):
        u, v = int(ei[0,i]), int(ei[1,i])
        if mn[i] > threshold and u in cset and v in cset:
            G.add_edge(u, v)
    if G.number_of_edges() == 0:
        return 1
    cliques = list(nx.find_cliques(G))
    return max(len(c) for c in cliques) if cliques else 1


# ── 5. Explainer ────────────────────────────────────────────────────────────

def make_explainer(model, n_classes, epochs=150):
    # Use multiclass_classification even for 2 classes; avoids return_type conflict
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=0.01),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )


# ── 6. Experiment: Multi-Size Clique Detection + Color Anchor Check ─────────

def run_multi_clique():
    print("\n" + "="*60)
    print("Experiment: Multi-Size Clique Detection + Color Anchor Check")
    print("="*60)

    (data, clique_groups,
     node_gt, node_clique_size) = build_multi_clique_dataset(
        n_nodes=200, er_p=0.02,
        clique_configs=[(3,8),(4,6),(5,5)]
    )
    cnt = {3:0, 4:0, 5:0}
    for cs, _ in clique_groups: cnt[cs] += 1
    print(f"  nodes={data.num_nodes}  edges={data.edge_index.size(1)//2}")
    print(f"  cliques: {cnt[3]}x(k=3)  {cnt[4]}x(k=4)  {cnt[5]}x(k=5)")
    n_cliq = int((data.y==1).sum()); n_base = data.num_nodes - n_cliq
    print(f"  clique nodes={n_cliq}  base nodes={n_base}")

    # Weighted loss to compensate for class imbalance
    cw = torch.tensor([1.0, n_base/n_cliq], dtype=torch.float)
    model = GCN(data.x.size(1), 64, 2, n_layers=2)
    print("  Training ...")
    model = train_model(model, data, n_epochs=600, wd=5e-5, class_weights=cw, verbose=True)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(1)
        acc  = (pred == data.y).float().mean().item()
        # per-class accuracy
        for cls, lbl in [(0,'base'),(1,'clique')]:
            mask = data.y == cls
            cacc = (pred[mask] == data.y[mask]).float().mean().item()
            print(f"  acc ({lbl}): {cacc:.4f}")
    print(f"  Final overall acc: {acc:.4f}")

    explainer = make_explainer(model, 2, epochs=150)
    res = {k: {'p':[],'r':[],'f1':[],'expl_sz':[],'color1':[],'color2':[]}
           for k in [3,4,5]}

    # Explain clique nodes: structural edge mask (as before) PLUS the color1
    # / color2 weight the explainer assigns to that same node's own
    # features. Both features are constant across every node, so if
    # GNNExplainer is faithful, both weights should be low and unrelated to
    # the class.
    print("  Explaining clique nodes ...")
    clique_nodes = []
    for csize, cnodes in clique_groups:
        for node_idx in cnodes[:3]:
            expl = explainer(data.x, data.edge_index, index=int(node_idx))
            p, r, f1 = edge_f1(expl.edge_mask, data.edge_index,
                                node_gt[node_idx])
            es = expl_clique_size(expl.edge_mask, data.edge_index, cnodes)
            color1_w, color2_w = expl.node_mask[node_idx].tolist()
            res[csize]['p'].append(p); res[csize]['r'].append(r)
            res[csize]['f1'].append(f1); res[csize]['expl_sz'].append(es)
            res[csize]['color1'].append(color1_w); res[csize]['color2'].append(color2_w)
            clique_nodes.append(node_idx)

    print()
    for k in [3,4,5]:
        r = res[k]
        print(f"  k={k} | P={np.mean(r['p']):.3f}±{np.std(r['p']):.3f}  "
              f"R={np.mean(r['r']):.3f}±{np.std(r['r']):.3f}  "
              f"F1={np.mean(r['f1']):.3f}±{np.std(r['f1']):.3f}  "
              f"expl_clique={np.mean(r['expl_sz']):.2f}/{k}")

    all_p  = [v for k in [3,4,5] for v in res[k]['p']]
    all_r  = [v for k in [3,4,5] for v in res[k]['r']]
    all_f1 = [v for k in [3,4,5] for v in res[k]['f1']]
    print(f"\n  Overall  P={np.mean(all_p):.3f}±{np.std(all_p):.3f}  "
          f"R={np.mean(all_r):.3f}±{np.std(all_r):.3f}  "
          f"F1={np.mean(all_f1):.3f}±{np.std(all_f1):.3f}")

    # Same check on a sample of non-clique nodes, so we can compare the
    # color1 / color2 weight GNNExplainer assigns across both classes.
    print("\n  Explaining a sample of non-clique nodes (anchor check) ...")
    rng = np.random.default_rng(SEED)
    base_pool = np.array([v for v in range(data.num_nodes)
                          if v not in set(clique_nodes) and data.y[v] == 0])
    base_sample = rng.choice(base_pool, size=min(50, len(base_pool)), replace=False)
    base_color1, base_color2 = [], []
    for node_idx in base_sample:
        expl = explainer(data.x, data.edge_index, index=int(node_idx))
        color1_w, color2_w = expl.node_mask[node_idx].tolist()
        base_color1.append(color1_w); base_color2.append(color2_w)

    clique_color1 = [v for k in [3,4,5] for v in res[k]['color1']]
    clique_color2 = [v for k in [3,4,5] for v in res[k]['color2']]
    print("\n  Color-feature mask weight sigma(m_F) "
          "(faithful => low and equal across classes):")
    print(f"    clique nodes     : color1={np.mean(clique_color1):.3f}  "
          f"color2={np.mean(clique_color2):.3f}")
    print(f"    non-clique nodes : color1={np.mean(base_color1):.3f}  "
          f"color2={np.mean(base_color2):.3f}")

    return {'acc': acc, 'res': res,
            'p': (np.mean(all_p), np.std(all_p)),
            'r': (np.mean(all_r), np.std(all_r)),
            'f1':(np.mean(all_f1),np.std(all_f1)),
            'color': {'clique_color1':  np.mean(clique_color1),
                      'clique_color2': np.mean(clique_color2),
                      'base_color1':    np.mean(base_color1),
                      'base_color2':   np.mean(base_color2)}}


# ── 7. Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cl = run_multi_clique()

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for k in [3,4,5]:
        r = cl['res'][k]
        print(f"  Clique k={k}:  "
              f"P={np.mean(r['p']):.3f}  R={np.mean(r['r']):.3f}  "
              f"F1={np.mean(r['f1']):.3f}  "
              f"avg_expl_clique={np.mean(r['expl_sz']):.2f}/{k}")
    print(f"  Clique overall:  "
          f"P={cl['p'][0]:.3f}  R={cl['r'][0]:.3f}  F1={cl['f1'][0]:.3f}")
    c = cl['color']
    print(f"  Color anchor:    clique color1={c['clique_color1']:.3f} "
          f"color2={c['clique_color2']:.3f}  |  non-clique "
          f"color1={c['base_color1']:.3f} color2={c['base_color2']:.3f}")