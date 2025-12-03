# Imports
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import copy
import numpy as np
import numpy as np
import csv
import json
import argparse
import random 
import os
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import RX, RY, RZ
from qulacs.gate import CNOT, CZ, SWAP
from qulacs.gate import X, Y, Z, H, T
from qulacs.gate import CZ, RY, RZ, merge
from qulacs import Observable

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from pprint import pprint
from typing import Optional, Union, Iterable, List, Tuple, Dict, Any

from scipy.optimize import minimize
from pathlib import Path

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parser
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
p = argparse.ArgumentParser(
        description="Extract the graph number and one selected cut (0–4) from a results file."
    )
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
def gen_maxcut(n: int, fc: bool, w_min: float, w_max: float, seed=None) -> np.ndarray:
    """
    Generate a random weighted graph for MaxCut as an n x n adjacency matrix W.

    Parameters
    ----------
    n : int
        Number of nodes.
    fc : bool
        If True, fully-connected with random weights in [w_min, w_max].
        If False, random Erdos-Renyi-ish with weights ±1.
    w_min, w_max : float
        Bounds for edge weights when fc=True.
    seed : any
        Seed for reproducibility (used in Python's random and numpy RNG).

    Returns
    -------
    W : np.ndarray, shape (n, n)
        Symmetric weight matrix with zeros on the diagonal.
    """
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    W = np.zeros((n, n), dtype=float)

    if fc:
        pw = np.linspace(w_min, w_max, num=21)   # same as before
        pw_list = pw.tolist()
        for i in range(n):
            for j in range(i + 1, n):
                w = py_rng.choice(pw_list)
                W[i, j] = W[j, i] = float(w)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                rd = py_rng.randint(0, 1)
                if rd <= 1/2:
                    rw = py_rng.randint(0, 1)
                    if rw <= 1/2:
                        W[i, j] = W[j, i] = 1.0
                    else:
                        W[i, j] = W[j, i] = -1.0

    return W

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# MAXCUT Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Qiskit Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------
def get_expval1(n, candidate_circuit, hamiltonian):
    state = QuantumState(n)
    candidate_circuit.update_quantum_state(state)
    return hamiltonian.get_expectation_value(state)
# -------------------------------------------------------------------------------------------------------
def cost_func_estimator(params, n, hamiltonian, pauli_params, pauli_weights, warm: bool, lukewarm : bool, pre_sol , eps, depth):
    ansatz = QAOA_hm_qulacs(n, params, pauli_params, pauli_weights, warm, lukewarm , pre_sol , eps, depth) 
    e = get_expval1(n, ansatz, hamiltonian)
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
    over β in [β_min, β_max] using scipy.optimize.minimize.
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

# Graph to Ising to Cost Hamiltonian
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def QUBO_to_Ising_weighted_graph(W: np.ndarray) -> Tuple[List[List[int]], List[float], float]:
    """
    Take a QUBO matrix W and return:
        - pauli_terms: list of indicator vectors for where Z acts (ZZ terms)
        - weights: corresponding coefficients
        - offset: constant shift
    """
    n = W.shape[0]
    pauli_terms: List[List[int]] = []
    weights: List[float] = []
    Q = W
    offset = np.triu(Q, 0).sum() / 2.0

    # Two-qubit Pauli-ZZ terms based on nonzero off-diagonal entries of Q
    for i in range(n - 1):
        for j in range(i + 1, n):
            if Q[i, j] != 0:  # If there's an edge (non-zero weight)
                term = np.zeros(n, dtype=int)
                term[i] = 1
                term[j] = 1
                pauli_terms.append(term.tolist())  # Pauli-ZZ between qubits i and j

                weight = Q[i, j] / 2.0
                weights.append(float(weight))

    return pauli_terms, weights, float(offset)
# -------------------------------------------------------------------------------------------------------
def ising_from_terms(
    pauli_terms: List[List[int]],
    weights: List[float],
    offset: float = 0.0,
    reverse_qubit_order: bool = True,
    build_qulacs_observable: bool = True,
):
    """
    Qulacs Observable using the same coefficients.
    """
    T = np.asarray(pauli_terms, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = T.shape[1]
    labels = []
    masks = []

    for row in T:
        mask = row > 0.5
        if reverse_qubit_order:
            mask_for_label = mask[::-1]
        else:
            mask_for_label = mask

        label = ''.join('Z' if b else 'I' for b in mask_for_label)
        labels.append(label)
        masks.append(mask)  # mask in the "natural" order of your pauli_terms

    #Qulacs Observable 
    qulacs_obs: Optional[Observable] = None
    if build_qulacs_observable:
        qulacs_obs = Observable(n)
        # Qulacs expects strings like "Z 0 Z 3" 
        for mask, coeff in zip(masks, w):
            pauli_ops = []
            for q, active in enumerate(mask):
                if active:
                    pauli_ops.append(f"Z {q}")
            if pauli_ops:
                op_str = " ".join(pauli_ops)
                qulacs_obs.add_operator(float(coeff), op_str)

    return qulacs_obs
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# QAOA ansatz build and solver (Qulacs)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cost_circ(circuit, n, pauli_terms: List[List[float]], weights: List[float], gamma):
    w = np.asarray(weights, dtype=float)
    for wk, term in zip(w, pauli_terms):
        idx = [i for i, b in enumerate(term) if b > 0.5]
        if len(idx) == 2:
            i, j = idx
            # RZZ(θ) = exp(-i θ ZZ / 2) => θ = 2 * gamma * w
            circuit.add_CNOT_gate(i, j)                   # First CNOT
            circuit.add_RZ_gate(j, -wk * gamma)       # RZ rotation on the target qubit (v)
            circuit.add_CNOT_gate(i, j) 
# -------------------------------------------------------------------------------------------------------
def mixer_circ(circuit, n, warm: bool, thetas, beta):
    if warm:
        for idx, theta in enumerate(thetas):
            circuit.add_RY_gate(idx, -theta)
            circuit.add_RZ_gate(idx, 2 * beta)
            circuit.add_RY_gate(idx, theta)
    else:
        for i in range(n):
            circuit.add_RX_gate(i, 2 * beta)
# -------------------------------------------------------------------------------------------------------
def init_circ(circuit, n, warm: bool, thetas):
    if warm:
        for idx, theta in enumerate(thetas):
            circuit.add_RY_gate(idx, -theta)
    else:
        for i in range(n):
            circuit.add_H_gate(i)
# -------------------------------------------------------------------------------------------------------
def QAOA_hm_qulacs(n, params, pauli_params, pauli_weights, warm: bool, lukewarm : bool, pre_sol , eps, depth):
    c_stars=[]
    for var in pre_sol:
        if var <= eps:
            c_stars.append(eps)     
        elif var >= 1-eps:    
            c_stars.append(1-eps)
        else:
            c_stars.append(var)
    thetas = [2 * np.arcsin(np.sqrt(c_star)) for c_star in c_stars]
    gammas = params[depth:]  # gamma[0],...,gamma[depth-1]
    betas  = params[:depth]  # beta[0],...,beta[depth-1]
    circuit = QuantumCircuit(n)
    #print("p",params,"g",gammas, "b",betas, flush = True)
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
    result = minimize(
        cost_func_estimator,
        parameters,
        args=(n, cost_operator, pauli_params, pauli_weights, warm, lukewarm, pre_sol , eps, depth),
        method="COBYLA",
        options={
        'rhobeg': 0.4,                  # ~10% of variable scale; tune 0.05–0.5
        'tol': 1e-4,                    # stopping tolerance; try 1e-3…1e-5 
        'maxiter': 100000,                         
    })
        
    print(result, flush = True )  
    print("COBYLA Optimization ended", result.x , flush = True)
    expval = cost_func_estimator(result.x, n, cost_operator, pauli_params, pauli_weights, warm, lukewarm, pre_sol , eps, depth)
    print("Expectation value of Hc after otimization:", expval, flush=True)
    #expval_id = get_expval1([np.pi/2, 0], circuit_w_ansz, cost_operator, backend)
    return expval
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
n=20                                                                    # Size of graph
eps = np.linspace(0,0.5,31)                                             # Reg. parameter
energies = []
# -------------------------------------------------------------------------------------------------------
G = gen_maxcut(n, True, -10, 10, graph)                                 # Graph generation, seeded 
pauli_terms, weights, offset = QUBO_to_Ising_weighted_graph(G)          # Generate Pauli terms and weights
print("p", pauli_terms, flush = True )
print("w", weights, flush = True )
H = ising_from_terms(pauli_terms, weights, 0)                           # Generate H (cost operator)
print("offset", offset, flush = True)
E_mc+=offset
# -------------------------------------------------------------------------------------------------------
depth = 1
# Grid search params
# -------------------------------------------------------------------------------------------------------
gamma_grid = np.linspace(-2*np.pi, 2*np.pi, 25)                         # 25 values from −2π to 2π
beta_probes = np.array([3*np.pi/4, 3*np.pi/8])                          # β ∈ {3π/4, 3π/8}

# Iteration w.r.t. reg. parameter
# -------------------------------------------------------------------------------------------------------
beta_star_per_gamma = []
E_star_per_gamma = []
for ep in eps:
    print(f"Regularization parameter: {ep} ---------------------------------------------------------")
    #########################
    beta_star_per_gamma1 = []
    E_star_per_gamma1 = []
    
    for g in gamma_grid:
        print(f"Gamma: {g} ----------------------------------")

        print(f"Beta: {beta_probes[0]} --------------------------")
        e1 = cost_func_estimator([beta_probes[0], g], n, H, pauli_terms, weights, True, False, cut , ep, depth)
        print(e1/E_mc, flush = True)
        
        print(f"Beta: {beta_probes[1]} --------------------------")
        e2 = cost_func_estimator([beta_probes[1], g], n, H, pauli_terms, weights, True, False, cut , ep, depth)
        print(e2/E_mc, flush = True)

        a, b = ab_from_energies_fixed_betas(e1, e2)
        print("Values of parameters:", a, b, flush = True)
        
        beta_star1, E_star_model1 = beta_star_from_ab_scipy(a, b, domain=(3*np.pi/8, 3*np.pi/4))
        print("beta_star (SciPy):", beta_star1, flush=True)

        E_star_true1 = cost_func_estimator([beta_star1, g], n, H, pauli_terms, weights, True, False, cut , ep, depth)
        print("E_star true (normalized):", E_star_true1/E_mc, flush=True)

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
    ew = solve_hm(n, True, False, H, pauli_terms, weights, cut, ep, 1, c, x_opt, [beta_star1, gamma_star1])
    print("h", ew/E_mc, flush=True)
    #print("h_id", ew_id/E_mc, flush=True)
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
out_dir = Path("/mnt/netapp1/Store_CESGA/home/cesga/jsouto/WS_GitHub/RoundQulacs/Results1")        # Out directory
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"results1_g{graph}_c{c}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



