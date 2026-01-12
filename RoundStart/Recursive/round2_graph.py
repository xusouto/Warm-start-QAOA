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
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parser
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MAXCUT Optimization Script")
parser.add_argument("--n", type = int, required= True, help = "Numer of graph")
parser.add_argument("--flag", type = str, required= True, help = "Numer of graph")
parser.add_argument("--iter", type = int, required= True, help = "Numer of graph")
parser.add_argument("--folder", type=Path, required=True,
                   help="Folder containing result JSON files.")
args = parser.parse_args()
n_ = args.n
flag = args.flag.lower() == "true"
iteration = args.iter
path = args.folder
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# MAXCUT problem 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def gen_maxcut(n, fc: bool, w_min, w_max, seed=None):

    py_rng = random.Random(seed)
    G = nx.Graph()
    G.add_nodes_from(np.arange(0, n, 1))

    pw = np.linspace(w_min, w_max, num=21)  
    pw_list = pw.tolist()                   

    elist = []
    if fc:
        for i in range(n):
            for j in range(i + 1, n):
                w = py_rng.choice(pw_list)  
                elist.append((i,j,w))
    else:
        for i in range(n):
            for j in range(i+1,n):
                rd=random.randint(0, 1)
                if rd <= 1/2:
                    rw=random.randint(0, 1)
                    if rw <= 1/2:
                        elist.append((i,j,1))
                    else:
                        elist.append((i,j,-1))

    G.add_weighted_edges_from(elist)

    colors = ["r" for node in G.nodes()]
    pos = nx.spring_layout(G)
    return G, colors, pos
# --------------------------------------------------------------------------------------------------
def graph_to_qp(G: nx.Graph) -> QuadraticProgram:
    nodes = list(sorted(G.nodes()))
    n = len(nodes)
    W = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = nodes.index(u), nodes.index(v)
        W[i, j] = W[j, i] = float(data.get("weight", 1.0))
    return Maxcut(W).to_quadratic_program()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Generate graph and MAXCUT with CPLEX
# -------------------------------------------------------------------------------------------------------
G, colors, pos = gen_maxcut(n_, flag, -10, 10, iteration)
qp = graph_to_qp(G)
print(f"Problem graph with {n_} nodes generated.", flush = True)
sol = CplexOptimizer().solve(qp)
x = np.asarray(sol.x, dtype=float)        # vector of variable values
x_bits = np.rint(x).astype(int).tolist()  # 0/1 integers
print(f"CPLEX solution for problem graph obtained.", flush = True)
G_py = nx.relabel_nodes(G, {u: int(u) for u in G.nodes()}, copy=True)
data = json_graph.node_link_data(G_py)   # now JSON-safe
# -------------------------------------------------------------------------------------------------------
# Out .jsons
# -------------------------------------------------------------------------------------------------------
out = {"graph": data, "sol": x_bits}
out_dir = Path(path) / "Graphs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"graph{n_}_{flag}_{iteration}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Graph data and cut stored in .json.", flush = True)
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



