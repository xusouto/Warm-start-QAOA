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
from qiskit.circuit import ParameterVector
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
parser.add_argument("--flag", type = str, required= True, help = "Numer of graph")
parser.add_argument("--iter", type = int, required= True, help = "Numer of graph")
args = parser.parse_args()
n_ = args.n
flag = args.flag.lower() == "true"
iteration = args.iter
# Graph info
with open(f"CS_RQAOA/Graphs/graph{n_}_{flag}_{iteration}.json", "r", encoding="utf-8") as f:
    data_graphs = json.load(f)   
G = json_graph.node_link_graph(data_graphs["graph"]) 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Graph to Ising to Cost Hamiltonian
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
            if Q[i][j] != 0:  # If there's an edge
                term = np.zeros(n)
                term[i] = 1
                term[j] = 1
                pauli_terms.append(term.tolist())  # Pauli-ZZ between qubits i and j
                
                weight = Q[i][j] / 2    # Weight for this edge
                weights.append(weight)  # Add to the weights list
  
    return pauli_terms, weights, offset
# -------------------------------------------------------------------------------------------------------
def graph_to_qp(G: nx.Graph) -> QuadraticProgram:
    nodes = list(sorted(G.nodes()))
    n = len(nodes)
    W = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = nodes.index(u), nodes.index(v)
        W[i, j] = W[j, i] = float(data.get("weight", 1.0))
    return Maxcut(W).to_quadratic_program()
# -------------------------------------------------------------------------------------------------------
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]
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

# Parameter initialization
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_custom_array_even_odd(length: int, lower1: float, upper1: float, lower2: float, upper2: float) -> np.ndarray:

    if length < 0:
        raise ValueError("Length must be non-negative.")
    if lower1 > upper1:
        raise ValueError("lower1 must be <= upper1.")
    if lower2 > upper2:
        raise ValueError("lower2 must be <= upper2.")

    even_val = np.random.uniform(lower1, upper1)
    odd_val  = np.random.uniform(lower2, upper2)
    arr = np.empty(length, dtype=float)
    arr[0::2] = even_val  
    arr[1::2] = odd_val   

    return arr
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Qiskit Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cost_func_estimator(params, ansatz, hamiltonian, backend):
    e = get_expval1(params, ansatz, hamiltonian, backend)
    return e
# -------------------------------------------------------------------------------------------------------
def prob_map_from_res(res) -> Dict[Tuple[int, ...], float]:
    pm: Dict[Tuple[int, ...], float] = defaultdict(float)
    for s in getattr(res, "raw_samples", []) or []:
        x = s.x.tolist() if hasattr(s.x, "tolist") else list(s.x)
        key = tuple(int(round(b)) for b in x)
        pm[key] += float(getattr(s, "probability", 0.0))
    # normalize (just in case)
    Z = sum(pm.values())
    if Z > 0:
        for k in list(pm.keys()):
            pm[k] /= Z
    return dict(pm)
# -------------------------------------------------------------------------------------------------------
def get_expval1(result, candidate_circuit, hamiltonian, backend):
    isa_hamiltonian = hamiltonian.apply_layout(candidate_circuit.layout) if getattr(candidate_circuit, "layout", None) else hamiltonian
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 500000
    job = estimator.run([(candidate_circuit, isa_hamiltonian, result)])
    result = job.result()[0]
    cost = float(result.data.evs)
    return cost
# -------------------------------------------------------------------------------------------------------
def get_probs(result, candidate_circuit, backend):
    sampler = Sampler(mode=backend)
    optimized_circuit = candidate_circuit.assign_parameters(result)
    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=int(1e5))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
    return final_distribution_bin
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
            circuit.rz(wk * gamma, j)          # RZ rotation on the target qubit (v)
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
    gammas = ParameterVector("gamma", depth)  # gamma[0],...,gamma[depth-1]
    betas  = ParameterVector("beta",  depth)  # beta[0],...,beta[depth-1]
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
def solve_hm(n, warm: bool, lukewarm: bool, cost_operator, pauli_params, pauli_weights, pre_sol, eps, depth, parameters):
    circuit_w_ansz = QAOA_hm_1(n, pauli_params, pauli_weights, warm, lukewarm, pre_sol , eps, depth) 
    backend = AerSimulator()
    backend.set_options(max_parallel_threads=5)
    backend.set_options(max_parallel_shots=5)
    result = minimize(
        cost_func_estimator,
        parameters,
        args=(circuit_w_ansz, cost_operator, backend),
        method="COBYLA",
        options={
        'rhobeg': 0.4,                  
        'tol': 1e-4,                    # stopping tolerance     
        'maxiter': 100000, 
    }) 
    print(result, flush = True )  
    print("COBYLA Optimization ended", result.x , flush = True)
    expval = get_expval1(result.x, circuit_w_ansz, cost_operator, backend)
    final_dist = get_probs(result.x, circuit_w_ansz, backend)
    print("Expectation value of Hc after otimization:", expval, flush=True)
    return expval, final_dist
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Cost operator generation
# -------------------------------------------------------------------------------------------------------
qp = graph_to_qp(G)
pauli_terms, weights, offset = QUBO_to_Ising_weighted_graph_nx(G)
H = ising_from_terms_qiskit(pauli_terms, weights, 0)
print("Curently solving cold-started QAOA instance", flush=True)
# -------------------------------------------------------------------------------------------------------
# Grid search
# -------------------------------------------------------------------------------------------------------
energies = []
depth = 1
eps = 0.25
beta_star1, gamma_star1 = create_custom_array_even_odd(2, -2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi )
print("γ*, β*:", gamma_star1, beta_star1, flush=True)
# -------------------------------------------------------------------------------------------------------
# Solve QAOA
# -------------------------------------------------------------------------------------------------------
cut = [random.randint(0, 1) for _ in range(n_)] 
ew, pwf = solve_hm(n_, False, False, H, pauli_terms, weights, cut, eps, 1, [beta_star1, gamma_star1])
print("Energy", ew, flush=True)
energies.append(ew)
# -------------------------------------------------------------------------------------------------------
# Out .jsons
# -------------------------------------------------------------------------------------------------------
out = {"n": n_,
       "flag": flag,
       "iter": iteration,
       "pwf": pwf,
       }
out_dir = Path("CS_RQAOA/Probabilities")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"probs{n_}_{flag}_{iteration}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



