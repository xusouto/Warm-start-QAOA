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
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parser
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Recursive Quantum Approximate Optimization Algorithm for MAXCUT (Qulacs)")
parser.add_argument("--n", type = int, required= True, help = "Number of nodes in graph")
parser.add_argument("--iter", type = int, required= True, help = "Number of graph")

args = parser.parse_args()
n_ = args.n
iteration = args.iter


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Problem graph generation 
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
# -------------------------------------------------------------------------------------------------------

# Cuts Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def graph_to_qp(G):
    nodes = list(range(G.number_of_nodes()))
    W = np.zeros((len(nodes), len(nodes)))
    for u, v, d in G.edges(data=True):
        W[u, v] = W[v, u] = float(d.get("weight", 1.0))
    return Maxcut(W).to_quadratic_program()
# -------------------------------------------------------------------------------------------------------
def cut_value(G, x_bits):
    part0 = {i for i,b in enumerate(x_bits) if b == 0}
    part1 = set(range(len(x_bits))) - part0
    w = 0.0
    for u, v, data in G.edges(data=True):
        w_ij = data.get("weight", 1.0)
        if (u in part0 and v in part1) or (u in part1 and v in part0):
            w += w_ij
    return w
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
G, colors, pos = gen_maxcut(n_, True, -10, 10, iteration)
# -------------------------------------------------------------------------------------------------------
# Calculation of exact solution with CPLEX and cut size
# -------------------------------------------------------------------------------------------------------
qp = graph_to_qp(G)
res = CplexOptimizer().solve(qp)
x_opt = np.array(res.x, dtype=int)
maxcut_size = cut_value(G, x_opt)

# --- After you compute x_opt and maxcut_size ---

results_path = Path(f"/Recursive/Results2/results_{n_}_{iteration}.json")

with open(results_path, "r", encoding="utf-8") as f:
    data = json.load(f)

data["exact"] = {
    "maxcut_value": float(maxcut_size),
    "solution": x_opt.astype(int).tolist(),
}

maxcut_val = float(maxcut_size)
skip_keys = {"n", "iter", "exact"}

for key, entry in list(data.items()):
    if key in skip_keys:
        continue
    if not isinstance(entry, dict):
        continue
    if "cut_value" not in entry:
        continue

    val = float(entry["cut_value"])
    entry["approx_ratio"] = (val / maxcut_val) if maxcut_val != 0.0 else None
    data[key] = entry

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Updated {results_path} with cut size for available cases only.", flush=True)
# -------------------------------------------------------------------------------------------------------
