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
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
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
p = argparse.ArgumentParser(description="Extract the graph number and one selected cut (0–4) from a results file.")
p.add_argument("file", help="Path to results file (e.g., results_g0.txt)")
p.add_argument("--cut", type=int, required=True, help="Cut index (0..4)")
args = p.parse_args()
with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)
c = args.cut
graph = data.get("graph")  # expected per your format
cut = data["cuts_list"][c]
E_mc = data.get("E_mc")
x_opt = data.get("x_opt")
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Graph generation 
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
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# MAXCUT Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
def cut_weight(G: nx.Graph, bits):
    """Sum of weights of edges crossing the cut defined by bits (0/1 per node)."""
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
# ---------------------------------------------------------------------------------------
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Qiskit Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_expval1(result, candidate_circuit, hamiltonian, backend):
    isa_hamiltonian = hamiltonian.apply_layout(candidate_circuit.layout) if getattr(candidate_circuit, "layout", None) else hamiltonian
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 1000000
    job = estimator.run([(candidate_circuit, isa_hamiltonian, result)])
    result = job.result()[0]
    cost = float(result.data.evs)
    return cost
# -------------------------------------------------------------------------------------------------------
def cost_func_estimator(params, ansatz, hamiltonian, backend):
    e = get_expval1(params, ansatz, hamiltonian, backend)
    return e
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parameter initialization
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _E_of_beta(a, b, beta):
    return a*np.sin(2*beta) + b*np.sin(4*beta)
# -------------------------------------------------------------------------------------------------------
def ab_from_energies_fixed_betas(E_beta1: float, E_beta2: float):
    a = -E_beta1
    b = -E_beta1 * (np.sqrt(2)/2) - E_beta2
    return a, b
# -------------------------------------------------------------------------------------------------------
def beta_star_from_ab_scipy(a: float, b: float, domain=(3*np.pi/8, 3*np.pi/4)):
    """
    Minimize the surrogate f(β) = a sin(2β) + b sin(4β)
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

    return circuit
# -------------------------------------------------------------------------------------------------------
def solve_hm(n, warm: bool, lukewarm: bool, cost_operator, pauli_params, pauli_weights, pre_sol, eps, depth, iteration, solution, parameters):
    circuit_w_ansz = QAOA_hm_1(n, pauli_params, pauli_weights, warm, lukewarm, pre_sol , eps, depth) 
    print("Number of parameters:", circuit_w_ansz.num_parameters, flush=True)
    print("Unique parameters:", circuit_w_ansz.parameters, flush=True)
    backend = AerSimulator()
    backend.set_options(max_parallel_threads=4)
    backend.set_options(max_parallel_shots=4)
    circuit_w_ansz = transpile(circuit_w_ansz, backend)  
    result = minimize(
        cost_func_estimator,
        parameters,
        args=(circuit_w_ansz, cost_operator, backend),
        method="COBYLA",
        options={
        'rhobeg': 0.4,                      
        'tol': 1e-4,
        'maxiter': 100000,                    # stopping tolerance        
    })
        
    print(result, flush = True )  
    print("COBYLA Optimization ended", result.x , flush = True)
    expval = get_expval1(result.x, circuit_w_ansz, cost_operator, backend)
    print("Expectation value of Hc after otimization:", expval, flush=True)
    expval_id = get_expval1([np.pi/2, 0], circuit_w_ansz, cost_operator, backend)
    return expval, expval_id
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Parameter setup
# -------------------------------------------------------------------------------------------------------
n=20                                                                    # Size of graph
eps = np.linspace(0,0.5,21)                                             # Reg. parameter
energies = []
depth = 1
# -------------------------------------------------------------------------------------------------------
# Graph generation
# -------------------------------------------------------------------------------------------------------
G, colors, pos = gen_maxcut(n, True, -10, 10, graph)                    # Graph generation, seeded 
pauli_terms, weights, offset = QUBO_to_Ising_weighted_graph_nx(G)       # Generate Pauli terms and weights
print("p", pauli_terms, flush = True )
print("w", weights, flush = True )
H = ising_from_terms_qiskit(pauli_terms, weights, 0)                    # Generate H (cost operator)
print("offset", offset, flush = True)
E_mc+=offset
# -------------------------------------------------------------------------------------------------------
# Grid search params
# -------------------------------------------------------------------------------------------------------
gamma_grid = np.linspace(-2*np.pi, 2*np.pi, 25)  # 25 values from −2π to 2π
beta_probes = np.array([3*np.pi/4, 3*np.pi/8])   # β ∈ {3π/4, 3π/8}
# -------------------------------------------------------------------------------------------------------
# Iteration w.r.t. reg. parameter
# -------------------------------------------------------------------------------------------------------
beta_star_per_gamma = []
E_star_per_gamma = []
backend = AerSimulator()
backend.set_options(max_parallel_threads=4)
backend.set_options(max_parallel_shots=4)
for ep in eps:
    print(f"Regularization parameter: {ep} ---------------------------------------------------------")
    #########################
    beta_star_per_gamma1 = []
    E_star_per_gamma1 = []
    
    for g in gamma_grid:
        print(f"Gamma: {g} ----------------------------------")

        qc_res_no_params = QAOA_hm_1(n, pauli_terms, weights, True, False, cut , ep, 1)
        qc_res_no_params = transpile(qc_res_no_params, backend)  

        print(f"Beta: {beta_probes[0]} --------------------------")
        e1 = get_expval1([beta_probes[0],g], qc_res_no_params, H, backend)
        print(e1/E_mc, flush = True)
        
        print(f"Beta: {beta_probes[1]} --------------------------")
        e2 = get_expval1([beta_probes[1],g], qc_res_no_params, H, backend)
        print(e2/E_mc, flush = True)

        a, b = ab_from_energies_fixed_betas(e1, e2)
        print("values of parameters", a, b, flush = True)
        
        beta_star1, E_star_model1 = beta_star_from_ab_scipy(a, b, domain=(3*np.pi/8, 3*np.pi/4))
        print("beta_star (from SciPy)", beta_star1, flush=True)

        E_star_true1 = get_expval1([beta_star1, g], qc_res_no_params, H, backend)
        print("E_star true (normalized)", E_star_true1/E_mc, flush=True)

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
    ew, ew_id = solve_hm(n, True, False, H, pauli_terms, weights, cut, ep, 1, c, x_opt, [beta_star1, gamma_star1])
    print("h", ew/E_mc, flush=True)
    print("h_id", ew_id/E_mc, flush=True)
    energies.append(ew/E_mc)
    print(f"------------------------------------------------------------------------------------")
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Out .jsons
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
eps_py = np.asarray(eps).tolist()                            # ndarray -> list
energies_py = [float(x) for x in energies] 
out = {
    "graph" : graph,
    "eps": eps_py,
    "cuts_list": cut,
    "Energies": energies_py,
}
out_dir = Path("/mnt/netapp1/Store_CESGA/home/cesga/jsouto/MAXCUT_multicore/Results")        # Out directory
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"results1_g{graph}_c{c}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



