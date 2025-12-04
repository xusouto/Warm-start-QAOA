# recursive_gw_step.py
# ---------------------------------------------------------------------
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

# ---------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Recursive GW correlation + reduction step")
parser.add_argument("--n", type=int, required=True, help="Number of nodes for current graph")
parser.add_argument("--flag", type=str, required=True, help="Flag used in filenames (True/False)")
parser.add_argument("--iter", type=int, required=True, help="Iteration / recursion level")
args = parser.parse_args()

n_ = args.n
flag = args.flag.lower() == "true"
iteration = args.iter


# ---------------------------------------------------------------------
# Load current graph
# ---------------------------------------------------------------------
graph_path = f"RGW/Graphs/graph{n_}_{flag}_{iteration}.json"
with open(graph_path, "r", encoding="utf-8") as f:
    data_graph = json.load(f)

G = json_graph.node_link_graph(data_graph["graph"])

cuts_path = f"RGW/Cuts/cuts{n_}_{flag}_{iteration}.json"
with open(cuts_path, "r", encoding="utf-8") as f:
    data_cuts = json.load(f)

cuts = np.asarray(data_cuts["cuts"], dtype=int)   # shape: (numcuts, n_bits)
numcuts = data_cuts.get("numcuts", cuts.shape[0])

if cuts.shape[0] != numcuts:
    print(f"WARNING: 'numcuts'={numcuts} but got {cuts.shape[0]} cuts in file.")

# Sanity check: number of bits vs number of nodes
nodes = sorted(G.nodes())
n_bits = cuts.shape[1]
if n_bits != len(nodes):
    raise ValueError(
        f"Mismatch: graph has {len(nodes)} nodes, "
        f"but cuts have {n_bits} bits per cut."
    )

print(f"Loaded graph with {len(nodes)} nodes and {cuts.shape[0]} GW cuts.", flush=True)

# ---------------------------------------------------------------------
# RQAOA-style reduction step (same as in your quantum correlation script)
# ---------------------------------------------------------------------
def rqaoa_reduce_step(G: nx.Graph, M: np.ndarray, nodes: list[int]):
    # Zero-out diagonal so we don't pick (i, i)
    M_off = M.copy()
    np.fill_diagonal(M_off, 0.0)

    # Find strongest correlation |M_ij|
    i_idx, j_idx = np.unravel_index(np.argmax(np.abs(M_off)), M_off.shape)
    i_node, j_node = nodes[i_idx], nodes[j_idx]
    alpha = 1 if M[i_idx, j_idx] >= 0 else -1

    G2 = G.copy()

    # Fold edges of i_node into j_node: w_{j,k} += alpha * w_{i,k}
    for k in list(G2.nodes()):
        if k in (i_node, j_node):
            continue
        w_ik = G2[i_node][k]["weight"] if G2.has_edge(i_node, k) else 0.0
        if w_ik != 0.0:
            w_jk = G2[j_node][k]["weight"] if G2.has_edge(j_node, k) else 0.0
            new_w = w_jk + alpha * w_ik
            if new_w == 0.0:
                if G2.has_edge(j_node, k):
                    G2.remove_edge(j_node, k)
            else:
                G2.add_edge(j_node, k, weight=new_w)

    # Remove i_node
    if G2.has_node(i_node):
        G2.remove_node(i_node)

    # New node list (maintain order except removed node)
    nodes2 = [u for u in nodes if u != i_node]
    return G2, nodes2, (i_node, j_node, alpha)

# ---------------------------------------------------------------------
# Build correlation matrix M from GW cuts
#   - bits 0/1 -> spins +1/-1 (same as your quantum code)
#   - M_ij = (1 / numcuts) * sum_{l} z_i^{(l)} z_j^{(l)}
# ---------------------------------------------------------------------
# bits in cuts: 0 -> +1, 1 -> -1
spins = 1 - 2 * cuts   # shape (numcuts, n), entries in {+1, -1}

# Average outer product
M = (spins.T @ spins) / float(spins.shape[0])  # shape (n, n)

print("Correlation matrix M from GW cuts:\n", M, flush=True)

# ---------------------------------------------------------------------
# Apply one recursive reduction step
# ---------------------------------------------------------------------
G_next, nodes_next, step = rqaoa_reduce_step(G, M, nodes)
print("Eliminated (i_node, j_node, alpha):", step, flush=True)
print("Remaining nodes:", nodes_next, flush=True)

# Ensure nodes are ints in the saved graph (if they aren't already)
relabel_map = {u: int(u) for u in G_next.nodes()}
G_p = nx.relabel_nodes(G_next, relabel_map, copy=True)

data_out_graph = json_graph.node_link_data(G_p)

# ---------------------------------------------------------------------
# Save new reduced graph + elimination info
# ---------------------------------------------------------------------
out = {
    "graph": data_out_graph,
    "eliminations": step,      # (i_node, j_node, alpha)
    "nodes_remaining": nodes_next,
}

out_dir = Path("RGW/Graphs")
out_dir.mkdir(parents=True, exist_ok=True)

# Decrease n_ by 1 for the reduced instance (same pattern as your quantum script)
out_path = out_dir / f"graph{n_ - 1}_{flag}_{iteration}.json"

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Saved reduced graph to {out_path}", flush=True)