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
from collections import defaultdict
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
with open(f"CS-RQAOA/Graphs/graph{n_}_{flag}_{iteration}.json", "r", encoding="utf-8") as f:
    data = json.load(f)   
G = json_graph.node_link_graph(data["graph"]) 
files = (
    [f"CS-RQAOA/Probabilities/probs{n_}_{flag}_{iteration}.json"]
    )
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_pwf(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pwf = data.get("pwf", {})
    return {str(k): float(v) for k, v in pwf.items()}  
# ---------------------------------------------------------------------------------------
def mean_pwf(files: List[Path]) -> Dict[str, float]:
    sums = defaultdict(float)
    counts = defaultdict(int)

    for p in files:
        try:
            pwf = load_pwf(p)
        except Exception as e:
            print(f"WARNING: skipping {p} ({e})")
            continue

        for k, v in pwf.items():
            sums[k] += v
            counts[k] += 1

    # average over only the files where the key appears
    return {k: (sums[k] / counts[k]) for k in sums.keys() if counts[k] > 0}
# ---------------------------------------------------------------------------------------
def rqaoa_reduce_step(G: nx.Graph, M: np.ndarray, nodes: list[int]):
    M_off = M.copy()
    np.fill_diagonal(M_off, 0.0)
    i_idx, j_idx = np.unravel_index(np.argmax(np.abs(M_off)), M_off.shape)
    i_node, j_node = nodes[i_idx], nodes[j_idx]
    alpha = 1 if M[i_idx, j_idx] >= 0 else -1

    G2 = G.copy()
    # fold edges of i into j: w_{jk} += alpha * w_{ik}
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

    if G2.has_node(i_node):
        G2.remove_node(i_node)

    nodes2 = [u for u in nodes if u != i_node]
    return G2, nodes2, (i_node, j_node, alpha)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Script
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Obtain correlation matrix
# ---------------------------------------------------------------------------------------
avg = mean_pwf(files)
print(avg, flush=True)
bitstrings = list(avg.keys())
probs = np.array([avg[b] for b in bitstrings])
probs /= probs.sum()  # normalize just in case
n = len(bitstrings[0])  # number of qubits/bits
M = np.zeros((n, n))
for b, p in zip(bitstrings, probs):
    # map bitstring to spin vector (+1 for '0', -1 for '1')
    z = np.array([1 if bit == '0' else -1 for bit in b])
    # add weighted outer product
    M += p * np.outer(z, z)
print("Correlation matrix:\n", M, flush=True)
# ---------------------------------------------------------------------------------------
# Remove nodes
# ---------------------------------------------------------------------------------------
nodes = sorted(G.nodes())                 # current node order used for reduction
G_next, nodes_next, step = rqaoa_reduce_step(G, M, nodes)
print("Eliminated:", step)  # (i_node, j_node, alpha)
print("Remaining nodes:", nodes_next)
G_p = nx.relabel_nodes(G_next, {u: int(u) for u in G.nodes()}, copy=True)
data = json_graph.node_link_data(G_p)   
# ---------------------------------------------------------------------------------------
# Out .jsons
# ---------------------------------------------------------------------------------------
out = {"graph": data, "eliminations": step, "nodes_remaining": nodes_next}
out_dir = Path("CS-RQAOA/Graphs")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"graph{n_-1}_{flag}_{iteration}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# ---------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

