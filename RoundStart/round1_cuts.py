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
parser = argparse.ArgumentParser(description="MAXCUT Optimization Script")
parser.add_argument("--graph", type = int, required= True, help = "Numer of graph")
args = parser.parse_args()
graph = args.graph
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
# ---------------------------------------------------------------------
def maxcut_obj(x, G):
    obj = 0
    for i, j, data in G.edges(data=True):
        weight = data.get("weight", 1.0) 
        if x[i] != x[j]:  
            obj -= weight  
    return obj
# -------------------------------------------------------------------------------------------------------
def _to_int_bits(x):
    """Coerce a list/array of 0/1 into clean ints."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [int(round(b)) for b in x]
# -------------------------------------------------------------------------------------------------------
def _cut_weight_from_x(G, x_bits):
    """Compute cut weight for bitstring x on graph G (assumes edge weights in 'weight')."""
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
# Weighted FC graph with 6 nodes from Egger's paper
# -------------------------------------------------------------------------------------------------------
n=20
G, colors, pos = gen_maxcut(n, True, -10, 10, graph)
# -------------------------------------------------------------------------------------------------------
# Calculation of exact solution with CPLEX and associated energy
# -------------------------------------------------------------------------------------------------------
qp = graph_to_qp(G)
print(f"Graph {graph} generated",flush=True)
res = CplexOptimizer().solve(qp)
x_opt = np.array(res.x, dtype=int)
E_mc = maxcut_obj(x_opt, G)
print("Energy calculated",flush=True)
print(f"Best cut {graph}", x_opt, flush = True)
# -------------------------------------------------------------------------------------------------------
# Calculation of initial cuts using GW
# -------------------------------------------------------------------------------------------------------
sol= GoemansWilliamsonOptimizer(n, sort_cuts=True, unique_cuts=True).solve(qp)
print("GW done",flush=True)
cuts = extract_gw_cuts(sol, top_k=10, best_m=5, G=G)
cuts_list = [d["x"] for d in cuts]
x_opt = np.asarray(x_opt).round().astype(int).ravel().tolist()
# -------------------------------------------------------------------------------------------------------
# Outputs saved in .json files
# -------------------------------------------------------------------------------------------------------
out = {
    "graph": graph,
    "cuts_list": cuts_list,
    "E_mc": E_mc,
    "x_opt": x_opt,
}
out_dir = Path("Cuts20")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"results_g{graph}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# ----------------------------------------------------s---------------------------------------------------------------------------------------------------------------------------
