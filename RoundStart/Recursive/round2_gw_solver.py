# Imports 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse
import json, os
import argparse, json
import networkx as nx

from pathlib import Path
from networkx.readwrite import json_graph
from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit.circuit.library import n_local
from qiskit_optimization.algorithms import MinimumEigenOptimizer, GoemansWilliamsonOptimizer
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_optimization.minimum_eigensolvers import NumPyMinimumEigensolver, SamplingVQE
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import CplexOptimizer
from collections import defaultdict
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_optimization.optimizers import SPSA, COBYLA
from docplex.mp.model import Model
from pprint import pprint
from typing import Optional, Union, Iterable, List, Tuple, Dict, Any
from networkx.readwrite import json_graph
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parser
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MAXCUT Optimization Script")
parser.add_argument("--n", type = int, required= True, help = "Numer of graph")
parser.add_argument("--flag", type = str, required= True, help = "Numer of graph")
parser.add_argument("--iter", type = int, required= True, help = "Numer of graph")
args = parser.parse_args()
n_ = args.n
flag = args.flag.lower() == "true"
iteration = args.iter
with open(f"Graphs/graph{n_}_{flag}_{iteration}.json", "r", encoding="utf-8") as f:
    data = json.load(f)   
G = json_graph.node_link_graph(data["graph"]) 
x_opt = data["sol"]
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Cuts Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def graph_to_qp(G: nx.Graph) -> QuadraticProgram:
    nodes = list(sorted(G.nodes()))
    n = len(nodes)
    W = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = nodes.index(u), nodes.index(v)
        W[i, j] = W[j, i] = float(data.get("weight", 1.0))
    return Maxcut(W).to_quadratic_program()
# -------------------------------------------------------------------------------------------------------
def cut_weight(G: nx.Graph, bits):
    x = np.asarray(bits, dtype=int).ravel()
    nodes = list(sorted(G.nodes()))
    idx = {v: i for i, v in enumerate(nodes)}
    w = 0.0
    for u, v, data in G.edges(data=True):
        if x[idx[u]] != x[idx[v]]:
            w += float(data.get("weight", 1.0))
    return w
# -------------------------------------------------------------------------------------------------------
def _to_int_bits(x):
    
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [int(round(b)) for b in x]
# -------------------------------------------------------------------------------------------------------
def _cut_weight_from_x(G, x_bits):
    
    part0 = {i for i,b in enumerate(x_bits) if b == 0}
    part1 = set(range(len(x_bits))) - part0
    w = 0.0
    for u, v, data in G.edges(data=True):
        w_ij = data.get("weight", 1.0)
        if (u in part0 and v in part1) or (u in part1 and v in part0):
            w += w_ij
    return w
# -------------------------------------------------------------------------------------------------------
def extract_gw_cuts(sol, top_k=10, best_m=5, G=None):
    """
    From a GoemansWilliamsonOptimizationResult `sol`, return:
      - top_k cuts (by fval descending),
      - best_m among those (again by fval).

    If G is provided (networkx graph), also include the actual cut weight on G.
    Returns dict with 'top_k' and 'best_m' lists of dicts.
    """
    samples = getattr(sol, "samples", None) or []
    # Sort by objective value descending (Max-Cut is a maximization)
    samples_sorted = sorted(samples, key=lambda s: s.fval, reverse=True)

    def pack(sample):
        x_bits = _to_int_bits(sample.x)
        rec = {
            "x": x_bits,
            "fval": float(sample.fval),
            "probability": getattr(sample, "probability", None),
            "status": getattr(sample, "status", None),
        }
        if G is not None:
            rec["cut_weight"] = _cut_weight_from_x(G, x_bits)
        return rec

    top_list = [pack(s) for s in samples_sorted[:top_k]]
    best_list = [pack(s) for s in samples_sorted[:min(best_m, len(samples_sorted), top_k)]]

    return best_list
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Calculation of initial cuts using GW
# -------------------------------------------------------------------------------------------------------
qp = graph_to_qp(G)
sol= GoemansWilliamsonOptimizer(n_, sort_cuts=True, unique_cuts=True).solve(qp)
best_cuts = extract_gw_cuts(sol, top_k=10, best_m=1)
best_cut = best_cuts[0]
x_gw = best_cut["x"]
print(x_gw , flush = True)
print(x_opt, flush = True)
w_opt = cut_weight(G, x_opt)
w_gw = cut_weight(G, x_gw)
print(w_gw , flush = True)
print(w_opt, flush = True)
cut_size = (w_gw / w_opt)
# -------------------------------------------------------------------------------------------------------
# Outputs saved in .json files
# -------------------------------------------------------------------------------------------------------
out = {"cuts_size": cut_size}
out_dir = Path("GW/Solutions")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"sol_graph{n_}_{flag}_{iteration}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


