# Imports
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import copy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
import argparse
import networkx as nx
import random
import os
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from qiskit.primitives import StatevectorSampler
from qiskit import QuantumCircuit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from pprint import pprint
from typing import Optional, Union, Iterable, List, Tuple, Dict, Any
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.optimizers import COBYLA
from qiskit_optimization.problems.variable import VarType
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import MinimumEigenOptimizer, GoemansWilliamsonOptimizer
from qiskit_optimization.utils import algorithm_globals 
from qiskit_optimization.optimizers import SPSA
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
from pathlib import Path
from qiskit.circuit.library import n_local
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit_optimization.minimum_eigensolvers import SamplingVQE
from qiskit_optimization import QuadraticProgram
from networkx.readwrite import json_graph
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parser
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="MAXCUT Optimization Script")
parser.add_argument("--n", type = int, required= True, help = "Numer of graph")
parser.add_argument("--t", type = int, required= True, help = "Numer of graph")
parser.add_argument("--flag", type = str, required= True, help = "Numer of graph")
parser.add_argument("--iter", type = int, required= True, help = "Numer of graph")
args = parser.parse_args()
n_init = args.n
n_final = args.t
flag = args.flag.lower() == "true"
iteration = args.iter
with open(f"Graphs/graph{n_final}_{flag}_{iteration}.json", "r", encoding="utf-8") as f:
    data = json.load(f)   
G1 = json_graph.node_link_graph(data["graph"]) 
remaining_nodes = data["nodes_remaining"]
with open(f"Graphs/graph{n_init}_{flag}_{iteration}.json", "r", encoding="utf-8") as f:
    data = json.load(f)   
G0 = json_graph.node_link_graph(data["graph"])
x_bits = data["sol"]
print(G0, flush = True)
eliminations_chain: list[tuple[int, int, int]] = []
graphs_dir = Path("Graphs")
for m in range(n_init - 1, n_final - 1, -1):   # N0-1, N0-2, ..., n_stop
    path_m = graphs_dir / f"graph{m}_{flag}_{iteration}.json"
    with open(path_m, "r", encoding="utf-8") as f:
        d = json.load(f)
    e = tuple(d["eliminations"])  # (i_node, j_node, alpha)
    eliminations_chain.append((int(e[0]), int(e[1]), int(e[2])))
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Generate graph
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
                w = py_rng.choice(pw_list)  # deterministic choice given seed
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
# ---------------------------------------------------------------------------------------
def graph_to_qp(G: nx.Graph) -> QuadraticProgram:
    nodes = list(sorted(G.nodes()))
    n = len(nodes)
    W = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = nodes.index(u), nodes.index(v)
        W[i, j] = W[j, i] = float(data.get("weight", 1.0))
    return Maxcut(W).to_quadratic_program()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Back-subtitution
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def back_substitute_full_assignment(
    eliminations: list[tuple[int, int, int]],
    remaining_nodes: list[int],
    x_small,
    all_nodes: list[int],
):

    x_small = np.asarray(x_small).ravel()
    if x_small.shape[0] != len(remaining_nodes):
        raise ValueError(
            f"x_small length {x_small.shape[0]} does not match remaining_nodes length {len(remaining_nodes)}"
        )

    # Accept either bits {0,1} or spins {+1,-1}; convert to spins z in {+1,-1}
    unique_vals = set(np.unique(x_small).tolist())
    if unique_vals.issubset({0, 1}):
        z_small = 1 - 2 * x_small.astype(int)   # 0->+1, 1->-1
    elif unique_vals.issubset({-1, 1}):
        z_small = x_small.astype(int)
    else:
        raise ValueError("x_small must be bits {0,1} or spins {-1,+1}")

    # Initialize spins for the remaining nodes
    z_full = {node: int(z_small[i]) for i, node in enumerate(remaining_nodes)}

    # Back-substitute in reverse order: for each eliminated (i <- alpha * j)
    for i_node, j_node, alpha in reversed(eliminations):
        if j_node not in z_full:
            raise KeyError(
                f"Back-substitution failed: spin for j_node {j_node} not assigned yet."
            )
        z_full[i_node] = int(alpha * z_full[j_node])

    # Convert spins to bits in the original node order
    x_full_bits = np.array([(1 - z_full[v]) // 2 for v in all_nodes], dtype=int)
    return x_full_bits
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# MAXCUT Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cut_weight(G: nx.Graph, bits):
    x = np.asarray(bits, dtype=int).ravel()
    nodes = list(sorted(G.nodes()))
    idx = {v: i for i, v in enumerate(nodes)}
    w = 0.0
    for u, v, data in G.edges(data=True):
        if x[idx[u]] != x[idx[v]]:
            w += float(data.get("weight", 1.0))
    return w
# ---------------------------------------------------------------------------------------
def graph_to_qp(G: nx.Graph) -> QuadraticProgram:
    nodes = list(sorted(G.nodes()))
    n = len(nodes)
    W = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = nodes.index(u), nodes.index(v)
        W[i, j] = W[j, i] = float(data.get("weight", 1.0))
    return Maxcut(W).to_quadratic_program()
# ---------------------------------------------------------------------------------------
def cplex_maxcut(G: nx.Graph):
    qp = graph_to_qp(G)
    res = CplexOptimizer().solve(qp)
    x_opt = np.array(res.x, dtype=int)
    w_opt = cut_weight(G, x_opt)
    return x_opt, w_opt
# ---------------------------------------------------------------------------------------
def percent_of_opt(G: nx.Graph, x_bits, w_opt=None):
    if w_opt is None:
        _, w_opt = cplex_maxcut(G)
    w = cut_weight(G, x_bits)
    return (w / w_opt) if w_opt != 0 else np.nan, w
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Get all nodes list
# ---------------------------------------------------------------------------------------
G0, colors, pos = gen_maxcut(n_init, flag, -10, 10, iteration)
all_nodes = list(sorted(G0.nodes()))
print(G0, flush = True)
# ---------------------------------------------------------------------------------------
# Calculate cuts
# ---------------------------------------------------------------------------------------
qp = graph_to_qp(G1)
solution= GoemansWilliamsonOptimizer(n_final, sort_cuts=True, unique_cuts=True).solve(qp)
x_small = np.array(solution.x, dtype=int)
x_full = back_substitute_full_assignment(eliminations_chain, remaining_nodes, x_small, all_nodes)
# ---------------------------------------------------------------------------------------
# Calculate cut-size
# ---------------------------------------------------------------------------------------
w_opt = cut_weight(G0, x_bits)
w_full = cut_weight(G0, x_full)
cut_size = (w_full / w_opt)
# ---------------------------------------------------------------------------------------
#  Out .jsons
# ---------------------------------------------------------------------------------------
out = {"cut_size": cut_size}
out_dir = Path("/mnt/netapp1/Store_CESGA/home/cesga/jsouto/WS_GitHub/RoundStart/Recursive/Solutions")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"graph{n_init}_{flag}_{iteration}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


