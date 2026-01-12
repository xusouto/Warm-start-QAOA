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
parser.add_argument("--cut", type = int, required= True, help = "Numer of graph")
args = parser.parse_args()
n_ = args.n
flag = args.flag.lower() == "true"
iteration = args.iter
cut_pos = args.cut
# Cut info
with open(f"WS_RQAOA/Cuts/cuts{n_}_{flag}_{iteration}.json", "r", encoding="utf-8") as f:
    data_cuts = json.load(f)   
cut = data_cuts["cuts"][cut_pos]
# Graph info
with open(f"WS_RQAOA/Graphs/graph{n_}_{flag}_{iteration}.json", "r", encoding="utf-8") as f:
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
            if Q[i][j] != 0:  # If there's an edge (weight > 0)
                term = np.zeros(n)
                term[i] = 1
                term[j] = 1
                pauli_terms.append(term.tolist())  # Pauli-ZZ between qubits i and j
                
                weight = Q[i][j] / 2  # Weight for this edge
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
def ab_from_energies_fixed_betas(E_beta1: float, E_beta2: float):
    a = -E_beta1
    b = -E_beta1 * (np.sqrt(2)/2) - E_beta2
    return a, b
# -------------------------------------------------------------------------------------------------------
def beta_opt_from_ab(a: float, b: float, g, qc_res_no_params, H, backend) -> float:

    candidates: List[float] = []
    disc = a*a + 32*b*b
    sqrt_disc = np.sqrt(disc)
    for x in [(-a + sqrt_disc)/(8*b), (-a - sqrt_disc)/(8*b)]:
        if -1 <= x <= 1:
            x = float(np.clip(x, -1.0, 1.0))
            theta = np.arccos(x)  # in [0,π]
            beta1 = 0.5*theta
            beta2 = np.pi - 0.5*theta
            candidates.extend([beta1 % np.pi, beta2 % np.pi])

    #E_star_good = get_expval1([beta_probes[0],g], qc_res_no_params, H, backend)
    #vals = [_model_energy(be, a, b) for be in candidates]
    vals = [get_expval1([be,g], qc_res_no_params, H, backend) for be in candidates]
    print("beta_stars---", candidates, flush = True)
    print("beta_stars vals---", vals, flush = True)
    return float(candidates[int(np.argmin(vals))]), float(vals[int(np.argmin(vals))])
# -------------------------------------------------------------------------------------------------------
def _E_of_beta(a, b, beta):
    return a*np.sin(2*beta) + b*np.sin(4*beta)
# -------------------------------------------------------------------------------------------------------
def beta_star_from_ab_scipy(a: float, b: float, domain=(3*np.pi/8, 3*np.pi/4)):
    """
    Minimize the surrogate f(β) = a sin(2β) + b sin(4β)
    over β in [β_min, β_max] using scipy.optimize.minimize.

    Returns
    -------
    beta_star : float
        β that minimizes the surrogate.
    E_star_model : float
        Minimum surrogate energy f(β*).
    """
    beta_min, beta_max = domain

    # initial guess: middle of the interval
    x0 = np.array([(beta_min + beta_max)/2.0], dtype=float)

    def model(beta_arr):
        beta = beta_arr[0]
        return _E_of_beta(a, b, beta)  # surrogate energy

    res = minimize(
        model,
        x0=x0,
        method="L-BFGS-B",
        bounds=[(beta_min, beta_max)],
        options={"ftol": 1e-10}
    )

    beta_star = float(res.x[0])
    E_star_model = float(res.fun)
    return beta_star, E_star_model
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Qiskit Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cost_func_estimator(params, ansatz, hamiltonian, backend):
    # If ansatz has a layout (because you transpiled), map the observable to it:
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
    # If ansatz has a layout (because you transpiled), map the observable to it:
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
    backend.set_options(max_parallel_threads=10)
    backend.set_options(max_parallel_shots=10)
    result = minimize(
        cost_func_estimator,
        parameters,
        args=(circuit_w_ansz, cost_operator, backend),
        method="COBYLA",
        options={
        'rhobeg': 0.4,                  # ~10% of variable scale; tune 0.05–0.5
        'tol': 1e-4,                    # stopping tolerance; try 1e-3…1e-5         
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
print("Curently solving warm-started QAOA instance with cut:", cut)
# -------------------------------------------------------------------------------------------------------
# Grid search
# -------------------------------------------------------------------------------------------------------
gamma_grid = np.linspace(-2*np.pi, 2*np.pi, 25)  # 25 values from −2π to 2π
beta_probes = np.array([3*np.pi/4, 3*np.pi/8])
beta_star_per_gamma = []
E_star_per_gamma = []
beta_star_per_gamma1 = []
E_star_per_gamma1 = []
energies = []
depth = 1
eps = 0.25
backend = AerSimulator()
backend.set_options(max_parallel_threads=10)
backend.set_options(max_parallel_shots=10)
#########################
for g in gamma_grid:
    print(f"Gamma: {g} ----------------------------------")
    qc_res_no_params = QAOA_hm_1(n_, pauli_terms, weights, True, False, cut , eps, depth)
    print(f"Beta: {beta_probes[0]} --------------------------")
    e1 = get_expval1([beta_probes[0],g], qc_res_no_params, H, backend)
    print(e1, flush = True)
    print(f"Beta: {beta_probes[1]} --------------------------")
    e2 = get_expval1([beta_probes[1],g], qc_res_no_params, H, backend)
    print(e2, flush = True)

    a, b = ab_from_energies_fixed_betas(e1, e2)
    print("values of parameters", a, b, flush = True)

    beta_star1, E_star_model1 = beta_star_from_ab_scipy(a, b, domain=(3*np.pi/8, 3*np.pi/4))
    print("beta_star (from SciPy surrogate)", beta_star1, flush=True)

    E_star_true1 = get_expval1([beta_star1, g], qc_res_no_params, H, backend)
    print("E_star true (normalized)", E_star_true1, flush=True)

    beta_star_per_gamma1.append(beta_star1)
    E_star_per_gamma1.append(E_star_true1)

print(f"End of grid search ----------------------------------")
print(f"-----------------------------------------------------------------------")

#########################
beta_star_per_gamma1 = np.array(beta_star_per_gamma1)
E_star_per_gamma1 = np.array(E_star_per_gamma1)

# Best (γ*, β*) 
best_idx1    = int(np.argmin(E_star_per_gamma1))   
gamma_star1  = float(gamma_grid[best_idx1])
beta_star1   = float(beta_star_per_gamma1[best_idx1])
E_star_best1 = float(E_star_per_gamma1[best_idx1])

print("Best γ*, β*:", gamma_star1, beta_star1)
print("Predicted minimum energy E*:", E_star_best1)
#########################

# Solve QAOA
ew, pwf= solve_hm(n_, True, False, H, pauli_terms, weights, cut, eps, 1, [beta_star1, gamma_star1])
print("Energy", ew, flush=True)
#print("h_id", ew_id/E_mc, flush=True)
energies.append(ew)
print(f"------------------------------------------------------------------------------------")
# -------------------------------------------------------------------------------------------------------
# Out .jsons
# -------------------------------------------------------------------------------------------------------
out = {"n": n_,
       "flag": flag,
       "iter": iteration,
       "cut_pos": cut_pos,  
       "pwf": pwf,
       }
out_dir = Path("WS_RQAOA/Probabilities")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"probs{n_}_{flag}_{iteration}_{cut_pos}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



