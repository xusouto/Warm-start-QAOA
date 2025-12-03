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
from qiskit.visualization import circuit_drawer
from qiskit.circuit import ParameterVector
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
p = argparse.ArgumentParser(description="Wamr-star QAOA with GW classical solution for different depths")
p.add_argument("--depth", type=int, required=True, help="Depth")
p.add_argument("--iter", type=int, required=True, help="Iteration")
args = p.parse_args()
depth = args.depth
iteration = args.iter
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Graph generation (n=6)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def gen_maxcut(n = 6):
    G = nx.Graph()
    G.add_nodes_from(np.arange(0, n, 1))
    elist = [(0,1,3), (0,2,3), (0,3,6), (0,4,9), (0,5,1), (1,2,4), (1,3,4), (1,4,-8), (1,5,4), (2,3,3), (2,4,-7), (2,5,1), (3,4,-7), (3,5,6), (4,5,-5)]
    G.add_weighted_edges_from(elist)
    colors = ["r" for node in G.nodes()]
    pos = nx.spring_layout(G)
    return G, colors, pos
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# MAXCUT Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def maxcut_obj(x, G):
    """
    Given a bitstring as a solution, this function returns
    the weighted number of edges shared between the two partitions
    of the graph.
    """
    obj = 0
    for i, j, data in G.edges(data=True):
        weight = data.get("weight", 1.0)  # Use the weight of the edge (default is 1 if not present)
        if x[i] != x[j]:  # The edge is cut between the two partitions
            obj -= weight  # Subtract the weight of the edge
    return obj
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parameter initialization
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_custom_array(length: int, lower1: float, upper1: float, lower2: float, upper2: float) -> np.ndarray:
    first_half_value = np.random.uniform(lower1, upper1)
    second_half_value = np.random.uniform(lower2, upper2)
    array = np.concatenate((
        np.full(length // 2, first_half_value),
        np.full(length // 2, second_half_value)
    ))
    return array
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# QAOA ansatz build and solver
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cost_circ(circuit, n, pauli_terms: List[List[float]], weights: List[float], gamma):
    w = np.asarray(weights, dtype=float)
    for wk, term in zip(w, pauli_terms):
        idx = [i for i, b in enumerate(term) if b > 0.5]
        if len(idx) == 2:
            i, j = idx
            # RZZ(θ) = exp(-i θ ZZ / 2) => θ = 2 * gamma * w
            circuit.cx(i, j)                   # First CNOT
            circuit.rz(wk * gamma, j)       # RZ rotation on the target qubit (v)
            circuit.cx(i, j) 
# -------------------------------------------------------------------------------------------------------
def mixer_circ(circuit, n, warm: bool, thetas, beta):
    if warm:
        for idx, theta in enumerate(thetas):
            circuit.ry(theta, idx)
            circuit.rz(-2 * beta, idx)
            circuit.ry(-theta, idx)
    else:
        for i in range(n):
            circuit.rx(2 * beta, i)
# -------------------------------------------------------------------------------------------------------
def init_circ(circuit, n, warm: bool, thetas):
    if warm:
        for idx, theta in enumerate(thetas):
            circuit.ry(theta, idx)
    else:
        for i in range(n):
            circuit.h(i)
# -------------------------------------------------------------------------------------------------------
def QAOA_hm_1(n, pauli_params, pauli_weights, warm: bool, lukewarm : bool, pre_sol , eps, depth):
    c_stars=[]
    for var in pre_sol:
        if var <= eps:
            c_stars.append(eps)     
        elif var >= 1-eps:    
            c_stars.append(1-eps)
        else:
            c_stars.append(var)
    thetas = [2 * np.arcsin(np.sqrt(c_star)) for c_star in c_stars]
    gammas = ParameterVector("gamma", depth)  
    betas  = ParameterVector("beta",  depth)  
    circuit = QuantumCircuit(n, name="QAOA")
    init_circ(circuit, n, warm, thetas)

    for j in range(depth):
        cost_circ(circuit, n, pauli_params, pauli_weights, gammas[j])
        if lukewarm:                                                  
            # warm start - normal mixer
            mixer_circ(circuit,n, False, thetas, betas[j])
        else:
            mixer_circ(circuit,n, warm, thetas, betas[j])
    circuit.measure_all()
    return circuit
# -------------------------------------------------------------------------------------------------------
def solve_hm(n, warm: bool, lukewarm: bool, cost_operator, pauli_params, pauli_weights, pre_sol, eps, depth, iteration, parameters):
    circuit_w_ansz = QAOA_hm_1(n, pauli_params, pauli_weights, warm, lukewarm, pre_sol , eps, depth) 
    backend = AerSimulator()
    result = minimize(
        cost_func_estimator,
        parameters,
        args=(circuit_w_ansz, cost_operator, backend),
        method="COBYLA",
        options={
        'rhobeg': 0.1,                  
        'tol': 1e-4,                             
    })
    print(result, flush = True )  
    print("COBYLA Optimization ended", result.x , flush = True)
    expval = get_expval1(result.x, circuit_w_ansz, cost_operator, backend)
    print("Expectation value of Hc after otimization:", expval, flush=True)
    target_list1=[1,0,0,0,0,1]                                                  # Solution 1
    target_string1 = ''.join(map(str, target_list1))
    target_list2=[0,1,1,1,1,0]                                                  # Solution 2
    target_string2 = ''.join(map(str, target_list2))
    optimized_circuit = circuit_w_ansz.assign_parameters(result.x)
    p1 = get_probs(optimized_circuit, backend, target_list1)
    p2 = get_probs(optimized_circuit, backend, target_list2)
    pw = p1 + p2
    return pw, expval
# -------------------------------------------------------------------------------------------------------
def get_probs(optimized_circuit, backend, pre_sol):
    sampler = Sampler(mode=backend)
    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=int(1e6))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
    bs = "".join("1" if float(b) > 0.5 else "0" for b in pre_sol)
    return float(final_distribution_bin.get(bs, 0.0))
# -------------------------------------------------------------------------------------------------------
def cost_func_estimator(params, ansatz, hamiltonian, backend):
    e = get_expval1(params, ansatz, hamiltonian, backend)
    return e
# -------------------------------------------------------------------------------------------------------
def get_expval1(result, candidate_circuit, hamiltonian, backend):
    isa_hamiltonian = hamiltonian.apply_layout(candidate_circuit.layout) if getattr(candidate_circuit, "layout", None) else hamiltonian
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 2000000
    job = estimator.run([(candidate_circuit, isa_hamiltonian, result)])
    result = job.result()[0]
    cost = float(result.data.evs)
    return cost
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Utils 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def QUBO_to_Ising_weighted_graph_nx(G: nx.Graph) -> Tuple[List[List[int]], List[float], float]:

    n = len(G.nodes)
    pauli_terms = []
    weights = []
    Q = nx.to_numpy_array(G)
    offset = np.triu(Q, 0).sum() / 2 
    
    # Two-qubit Pauli-ZZ terms based on edges in the graph 
    for i in range(n - 1):
        for j in range(i + 1, n):
            if Q[i][j] != 0:  
                term = np.zeros(n)
                term[i] = 1
                term[j] = 1
                pauli_terms.append(term.tolist())  # Pauli-ZZ between qubits i and j  
                weight = Q[i][j] / 2  
                weights.append(weight)  
  
    return pauli_terms, weights, offset
# -------------------------------------------------------------------------------------------------------
def ising_from_terms_qiskit(pauli_terms: List[List[float]], weights: List[float], offset: float = 0.0, reverse_qubit_order: bool = True):
    T = np.asarray(pauli_terms, dtype=float)
    w = np.asarray(weights, dtype=float)
    labels = []
    for row in T:
        mask = row > 0.5  
        if reverse_qubit_order:
            mask = mask[::-1]
        label = ''.join('Z' if b else 'I' for b in mask)
        labels.append(label)
    op = SparsePauliOp.from_list(list(zip(labels, w.astype(complex))))
    if offset != 0.0:
        op += SparsePauliOp.from_list([('I' * n, complex(offset))])
    return op
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAXCUT problem generation from paper
# -------------------------------------------------------------------------------------------------------
n = 6
G, colors, pos = gen_maxcut(n)                                                                            # Unique graph displayed in paper
E_mc = maxcut_obj([0,1,1,1,1,0], G)                                                                       # Known MAXCUT [1,0,0,0,0,1] also valid
pauli_terms, weights, offset = QUBO_to_Ising_weighted_graph_nx(G)
H = ising_from_terms_qiskit(pauli_terms, weights, 0)
# -------------------------------------------------------------------------------------------------------
# Solve MAXCUT with WS-QAOA
# -------------------------------------------------------------------------------------------------------
backend = AerSimulator()
parms = create_custom_array(2*depth, 0, 2*np.pi, -2*np.pi, 2*np.pi)
pw, ew= solve_hm(n, True, False, H, pauli_terms, weights, [0, 0, 1, 1, 1, 1], 0.25, depth, iteration, parms)
# -------------------------------------------------------------------------------------------------------
# Out .jsons
# -------------------------------------------------------------------------------------------------------
out = {
    "depth": depth,
    "iteration": iteration,
    "prob": pw,
    "ener_norm": ew,
    "E_mc": E_mc}
out_dir = Path("/mnt/netapp1/Store_CESGA/home/cesga/jsouto/WS_GitHub/RoundStart/Results3")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"results_d{depth}_i{iteration}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
