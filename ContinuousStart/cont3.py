# Imports
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import copy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
import argparse
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from docplex.mp.model import Model
from qiskit.primitives import StatevectorSampler
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit.circuit import Parameter
from qiskit_optimization import QuadraticProgram
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit import ParameterVector
from qiskit_optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from pprint import pprint
from typing import Optional, Union, Iterable, List, Tuple, Dict, Any
from qiskit.circuit.library import QAOAAnsatz
from qiskit.visualization import circuit_drawer
from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.optimizers import COBYLA
from qiskit_optimization.problems.variable import VarType
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.utils import algorithm_globals 
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
from typing import Any
from typing import Tuple, Optional
Array = Any
Tensor = Any
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parser
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Quantum Finalcial Portfolio Optimization Script")
parser.add_argument("--time", type = int, required= True, help = "Depth of circuit")
parser.add_argument("--steps", type=int, required=True, help="Iteration number")
args = parser.parse_args()
T = args.time
N = args.steps
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Quadratic problem 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_problem(mu: np.array, sigma: np.array, total, q, lam) -> QuadraticProgram:
    mdl = Model()
    x = [mdl.binary_var("x%s" % i) for i in range(len(sigma))]
    objective = mdl.sum([mu[i] * x[i] for i in range(len(mu))])
    objective -= q * mdl.sum(
        [sigma[i, j] * x[i] * x[j] for i in range(len(mu)) for j in range(len(mu))]
    )
    sumx = mdl.sum(x)
    penalty_expr = lam * (sumx - total) * (sumx - total)
    objective -= penalty_expr
    mdl.maximize(objective)
    cost = mdl.sum(x)
    mdl.add_constraint(cost == total)
    qp = from_docplex_mp(mdl)
    return qp
def create_problem1(mu, Sigma, total, q, lam):
    n = len(mu)
    mdl = Model()
    x = [mdl.binary_var(name=f"x{i}") for i in range(n)]
    sumx = mdl.sum(x)

    # soft budget version (as in the paper)
    mdl.maximize(
        mdl.sum(mu[i]*x[i] for i in range(n))
        - q * mdl.sum(Sigma[i,j]*x[i]*x[j] for i in range(n) for j in range(n))
        - lam * (sumx - total) * (sumx - total)
    )
    return from_docplex_mp(mdl)
# -------------------------------------------------------------------------------------------------------
def relax_problem(problem) -> QuadraticProgram:
    """Change all variables to continuous."""
    relaxed_problem = copy.deepcopy(problem)
    for variable in relaxed_problem.variables:
        variable.vartype = VarType.CONTINUOUS
    return relaxed_problem
# -------------------------------------------------------------------------------------------------------
def simulate_portfolio_gbm(n: int = 6, N: int = 250, *, mu_range = (-0.05, 0.05), sigma_range = (-0.20, 0.20), seed: int):
    """
    Simulate n asset price paths via GBM for N days then return (mu, Sigma) 
    """
    rng = np.random.default_rng(seed)
    mu_i = rng.uniform(mu_range[0], mu_range[1], size=n)     
    sigma_i = rng.uniform(sigma_range[0], sigma_range[1], size=n)     

    # Brownian motion increments
    z = rng.standard_normal(size=(n, N))        # independent per asset/day
    W = np.cumsum(z, axis=1) / np.sqrt(N)           

    # Build S_{i,k} using S_{i,0}=1 and the closed-form GBM solution
    k_over_N = (np.arange(1, N+1) / N)[None, :]     
    drift    = (mu_i[:, None] - 0.5 * sigma_i[:, None]**2) * k_over_N
    diff     = sigma_i[:, None] * W
    S = np.empty((n, N+1))
    S[:, 0] = 1.0
    S[:, 1:] = np.exp(drift + diff)              

    # Daily returns 
    R = S[:, 1:] / S[:, :-1] - 1.0            

    # Mean return per asset 
    mu_vec = R.mean(axis=1)                       

    # Covariance across assets of daily returns 
    Sigma = np.cov(R, rowvar=True, bias=False)       

    return mu_vec, Sigma
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# QAOA Ansatz functions and solver
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def warm_start_thetas(pre_sol, eps=0.25):
    c_stars=[]
    for var in pre_sol:
        if var <= eps:
            c_stars.append(eps)     
        elif var >= 1-eps:    
            c_stars.append(1-eps)
        else:
            c_stars.append(var)
    thetas = [2 * np.arcsin(np.sqrt(c_star)) for c_star in c_stars]
    return thetas
# -------------------------------------------------------------------------------------------------------
def apply_equal_superposition_mixer(circ, qubits, beta):
    for q in qubits:
        circ.rx(-2.0*beta, q)
# -------------------------------------------------------------------------------------------------------
def apply_warm_start_mixer(circ, qubits, thetas, beta):
    for q, theta in zip(qubits, thetas):
        circ.ry(-theta, q)
        circ.rz(-2.0*beta, q)
        circ.ry(theta, q)
# -------------------------------------------------------------------------------------------------------
def prepare_equal_superposition(circ, qubits):
    for q in qubits:
        circ.h(q)
# -------------------------------------------------------------------------------------------------------
def prepare_warm_start_state(circ, qubits, thetas):
    for q, theta in zip(qubits, thetas):
        circ.ry(theta, q)
# -------------------------------------------------------------------------------------------------------
def cost(circ, pauli_terms: List[List[float]], weights: List[float], gamma) -> None:
    w = np.asarray(weights, dtype=float)
    for wk, term in zip(w, pauli_terms):
        idx = [i for i, b in enumerate(term) if b > 0.5]
        if len(idx) == 1:
            q = idx[0]
            # RZ(θ) = exp(-i θ Z / 2)  => θ = 2 * gamma * w
            circ.rz(wk * gamma, q)
        elif len(idx) == 2:
            i, j = idx
            # RZZ(θ) = exp(-i θ ZZ / 2) => θ = 2 * gamma * w
            circ.rzz(wk * gamma, i, j)
# -------------------------------------------------------------------------------------------------------
def anneal_circuit_from_qubo(n, Q, T, N, mixer="warm", c_star=None, eps=0.0):

    pauli_terms, weights, offset = QUBO_to_Ising(Q)
    qr = QuantumRegister(n, "q")
    circ = QuantumCircuit(qr, name="WS-anneal")

    qubits = list(range(n))
    delta_t = T / N

    # initial state
    if mixer == "equal":
        prepare_equal_superposition(circ, qubits)
        thetas = None
    elif mixer == "warm":
        if c_star is None:
            raise ValueError("Warm-start mixer requires c_star.")
        thetas = warm_start_thetas(c_star, eps=eps)
        prepare_warm_start_state(circ, qubits, thetas)
    else:
        raise ValueError("mixer must be 'equal' or 'warm'")

    # Trotter loop
    for k in range(N+1):
        beta_k  = 2.0 * delta_t * (1.0 - k / N)   # as in the text
        gamma_k = 2.0 * (k / N) * delta_t
        
        # cost evolution
        cost(circ, pauli_terms, weights, gamma_k)
        
        # mixer evolution
        if mixer == "equal":
            apply_equal_superposition_mixer(circ, qubits, beta_k)
        else:
            apply_warm_start_mixer(circ, qubits, thetas, beta_k)

    return circ
# -------------------------------------------------------------------------------------------------------
def solve_hm(n, Q, T, N, mixer, cost_operator, pre_sol, eps, solution):
    circuit_w_ansz = anneal_circuit_from_qubo(n, Q, T, N, mixer, pre_sol, eps)
    backend = AerSimulator()
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    candidate_circuit = pm.run(circuit_w_ansz)
    expval = get_expval1(candidate_circuit, cost_operator, backend)
    print("Expectation value of Hc after otimization:", expval, flush=True)
    return expval
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Qiskit Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]
# -------------------------------------------------------------------------------------------------------
def _bs_from_pre_sol(pre_sol) -> str:
    return "".join("1" if float(b) > 0.5 else "0" for b in pre_sol)
def get_probs1(circuit: QuantumCircuit, result, pre_sol):
    """
    Exact P(pre_sol) using StatevectorSampler, with a measured copy of the circuit
    to avoid the 'no classical registers' warning.
    `result` can be OptimizeResult (.x) or array-like.
    """
    params = getattr(result, "x", result)
    bound = circuit.assign_parameters(params).copy()

    # Add classical bits + measurements to avoid the warning (on the copy only)
    bound.measure_all(add_bits=True)

    bs = _bs_from_pre_sol(pre_sol)
    n = bound.num_qubits
    if len(bs) != n:
        raise ValueError(f"pre_sol length ({len(bs)}) != circuit qubits ({n})")

    sampler = StatevectorSampler()
    res = sampler.run([bound]).result()

    # Case A: classic SamplerResult with quasi_dists (int keys, little-endian)
    qd = getattr(res, "quasi_dists", None)
    if qd is not None:
        return float(qd[0].get(int(bs[::-1], 2), 0.0))

    # Case B: PrimitiveResult style with bitstring probabilities
    try:
        probs_bin = res[0].data.meas.get_probabilities()  # {'0101': p, ...}
        return float(probs_bin.get(bs, 0.0))
    except Exception:
        # Fallback: pure statevector (shouldn't be hit with StatevectorSampler)
        sv = Statevector.from_instruction(bound.remove_final_measurements(inplace=False))
        return float(abs(sv.data[int(bs[::-1], 2)])**2)
def get_probs(circuit, backend, pre_sol):
    optimized_circuit = circuit
    sampler = Sampler(mode=backend)
    pub = (optimized_circuit,)
    job = sampler.run([pub], shots=int(1e4))
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
    bs = "".join("1" if float(b) > 0.5 else "0" for b in pre_sol)
    return float(final_distribution_bin.get(bs, 0.0))
# -------------------------------------------------------------------------------------------------------
def get_expval1(candidate_circuit, hamiltonian, backend):
    # If ansatz has a layout (because you transpiled), map the observable to it:
    estimator = StatevectorEstimator()
    job = estimator.run([(candidate_circuit, hamiltonian)])
    result = job.result()[0]
    cost = float(result.data.evs)
    return cost
def get_expval(candidate_circuit, hamiltonian, backend):
    # If ansatz has a layout (because you transpiled), map the observable to it:
    isa_hamiltonian = hamiltonian.apply_layout(candidate_circuit.layout) if getattr(candidate_circuit, "layout", None) else hamiltonian
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = 500000
    job = estimator.run([(candidate_circuit, isa_hamiltonian)])
    result = job.result()[0]
    cost = float(result.data.evs)
    return cost
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Problem Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def QUBO_to_Ising(Q: Tensor) -> Tuple[Tensor, List[float], float]:
    """
    Cnvert the Q matrix into a the indication of pauli terms, the corresponding weights, and the offset.
    The outputs are used to construct an Ising Hamiltonian for QAOA.
    """

    # input is n-by-n symmetric numpy array corresponding to Q-matrix
    # output is the components of Ising Hamiltonian

    n = Q.shape[0]

    # square matrix check
    if Q[0].shape[0] != n:
        raise ValueError("Matrix is not a square matrix.")

    offset = (
        np.triu(Q, 0).sum() / 2
    )  # Calculate the offset term of the Ising Hamiltonian
    pauli_terms = []  # List to store the Pauli terms
    weights = (
        -np.sum(Q, axis=1) / 2
    )  # Calculate the weights corresponding to each Pauli term

    for i in range(n):
        term = np.zeros(n)
        term[i] = 1
        pauli_terms.append(
            term.tolist()
        )  # Add a Pauli term corresponding to a single qubit

    for i in range(n - 1):
        for j in range(i + 1, n):
            term = np.zeros(n)
            term[i] = 1
            term[j] = 1
            pauli_terms.append(
                term.tolist()
            )  # Add a Pauli term corresponding to a two-qubit interaction

            weight = (
                Q[i][j] / 2
            )  # Calculate the weight for the two-qubit interaction term
            weights = np.concatenate(
                (weights, weight), axis=None
            )  # Add the weight to the weights list

    return pauli_terms, weights, offset
# -------------------------------------------------------------------------------------------------------
def QUBO_from_portfolio(cov: Array, mean: Array, q: float, B: int, t: float) -> Tensor:
    """
    Convert portfolio parameters to a Q matrix
    """
    n = cov.shape[0]
    R = np.diag(mean)
    S = np.ones((n, n)) - 2 * B * np.diag(np.ones(n))

    Q = q * cov - R + t * S
    return Q
# -------------------------------------------------------------------------------------------------------
def ising_from_terms_qiskit(
    pauli_terms: List[List[float]],
    weights: List[float],
    offset: float = 0.0,
    reverse_qubit_order: bool = True,
) -> SparsePauliOp:
    """
    Build H = sum_k w_k * Z^{(term_k)} + offset * I from:
      - pauli_terms: list of 0/1 masks (floats ok) length n each
      - weights:    list/array of coefficients (same length as pauli_terms)
    Returns a Qiskit SparsePauliOp.
    """
    T = np.asarray(pauli_terms, dtype=float)
    w = np.asarray(weights, dtype=float)

    if T.ndim != 2:
        raise ValueError("pauli_terms must be a 2D list/array of shape (m, n).")
    m, n = T.shape
    if w.shape[0] != m:
        raise ValueError(f"weights length {w.shape[0]} != number of terms {m}.")

    labels = []
    for row in T:
        mask = row > 0.5  # accept 1.0 or near-1.0; ignore tiny noise
        if reverse_qubit_order:
            mask = mask[::-1]
        label = ''.join('Z' if b else 'I' for b in mask)
        labels.append(label)

    op = SparsePauliOp.from_list(list(zip(labels, w.astype(complex))))
    if offset != 0.0:
        op += SparsePauliOp.from_list([('I' * n, complex(offset))])
    return op
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Energy calculation
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def ground_energy_from_terms(
    pauli_terms: List[List[int]],
    weights: List[float],
    offset: float = 0.0,
) -> Tuple[float, List[int]]:
    """
    Compute E0 and one minimizing bitstring for an Ising-type Hamiltonian.
    """
    pauli_terms = [list(t) for t in pauli_terms]
    weights = np.asarray(weights, dtype=float)
    n = len(pauli_terms[0])
    m = len(pauli_terms)

    # Generate all bitstrings (rows) and map to spins z in {+1,-1}
    num = 1 << n
    a = np.arange(num, dtype=np.uint64)
    # bits[:, i] is bit of qubit i (LSB = qubit 0)
    shifts = np.arange(n, dtype=np.uint64)           # CRITICAL: unsigned dtype
    bits = ((a[:, None] >> shifts) & np.uint64(1)).astype(np.int8)
    z = 1 - 2 * bits  # 0->+1, 1->-1

    # Precompute indices for each term
    term_indices = [np.flatnonzero(t) for t in pauli_terms]

    # Accumulate energies
    E = np.full(num, offset, dtype=float)
    for w, idx in zip(weights, term_indices):
        if len(idx) == 1:
            E += w * z[:, idx[0]]
        elif len(idx) == 2:
            i, j = int(idx[0]), int(idx[1])
            E += w * (z[:, i] * z[:, j])
        else:
            # If you ever had higher-order Z terms, this still works:
            # E += w * np.prod(z[:, idx], axis=1)
            raise ValueError("Only Z and ZZ terms are supported.")

    # Ground energy and a minimizer
    k_min = int(np.argmin(E))
    E0 = float(E[k_min])
    ground_bits = bits[k_min].tolist()  # [b0, b1, ..., b_{n-1}]
    return E0, ground_bits
# -------------------------------------------------------------------------------------------------------
def print_Q_cost(Q, wrap=False, reverse=False):
    n_stocks = len(Q)
    states = []
    for i in range(2**n_stocks):
        a = f"{bin(i)[2:]:0>{n_stocks}}"
        n_ones = 0
        for j in a:
            if j == "1":
                n_ones += 1
        states.append(a)

    cost_dict = {}
    for selection in states:
        x = np.array([int(bit) for bit in selection])
        cost_dict[selection] = np.dot(x, np.dot(Q, x))
    cost_sorted = dict(sorted(cost_dict.items(), key=lambda item: item[1]))
    if reverse == True:
        cost_sorted = dict(
            sorted(cost_dict.items(), key=lambda item: item[1], reverse=True)
        )
    num = 0
    print("\n-------------------------------------")
    print("    selection\t  |\t  cost")
    print("-------------------------------------")
    for k, v in cost_sorted.items():
        print("%10s\t  |\t%.4f" % (k, v))
        num += 1
        if (num >= 8) & (wrap == True):
            break
    print("     ...\t  |\t  ...")
    print("-------------------------------------")
# -------------------------------------------------------------------------------------------------------
def energy_from_terms(
    bits: List[int],
    pauli_terms: List[List[float]],
    weights: List[float],
    offset: float = 0.0,
) -> float:
    """
    Compute E(bits) for H = sum_k w_k * Z^{(term_k)} + offset * I.
    bits is a list like [b0, b1, ..., b_{n-1}] with b in {0,1}.
    """
    z = 1 - 2 * np.asarray(bits, dtype=int)  # 0->+1, 1->-1
    E = float(offset)
    for w, term in zip(weights, pauli_terms):
        idx = np.flatnonzero(np.asarray(term, dtype=float) > 0.5)
        if len(idx) == 0:
            # identity term (not expected here, but handle gracefully)
            E += w
        elif len(idx) == 1:
            E += w * z[idx[0]]
        elif len(idx) == 2:
            E += w * (z[idx[0]] * z[idx[1]])
        else:
            # Supports higher-order Z products too:
            E += w * np.prod(z[idx])
    return float(E)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Financial Portfolio parameters
# -------------------------------------------------------------------------------------------------------
n = 6
seed = 10
mu, sigma = simulate_portfolio_gbm(n, 250, mu_range = (-0.05, 0.05), sigma_range = (-0.20, 0.20),  seed = seed)
B = 3                                                             # Number of selected assets
l = 3                                                             # Lambda for penalty term
q = 2                                                             # Risk-return trade-off
# -------------------------------------------------------------------------------------------------------
# Creation of problem and cost operator
# -------------------------------------------------------------------------------------------------------
Q = QUBO_from_portfolio(sigma, mu, q, B, l)
print("Q", Q, flush=True)
portfolio_pauli_terms, portfolio_weights, portfolio_offset = QUBO_to_Ising(Q)
print_Q_cost(Q, wrap=True)
print(portfolio_offset)
print(ground_energy_from_terms(portfolio_pauli_terms,portfolio_weights,portfolio_offset))
H = ising_from_terms_qiskit(portfolio_pauli_terms, portfolio_weights, portfolio_offset)
# -------------------------------------------------------------------------------------------------------
# Classical solution using CPLEX
# -------------------------------------------------------------------------------------------------------
qp = create_problem1(mu, sigma, B, q, l)
print("qp", qp, flush=True)
sol= CplexOptimizer().solve(qp)
c_stars = sol.samples[0].x
print(c_stars, flush=True)
# -------------------------------------------------------------------------------------------------------
# Ground state energy 
# -------------------------------------------------------------------------------------------------------
E0 = energy_from_terms(c_stars, portfolio_pauli_terms, portfolio_weights)
print("Energy:", E0)
print("Energy + offset:", E0+portfolio_offset)
# -------------------------------------------------------------------------------------------------------
# Relax problem and solve classically using CPLEX
# -------------------------------------------------------------------------------------------------------
qp_r = relax_problem(qp)
sol_r= CplexOptimizer().solve(qp_r)
c_stars_r = sol_r.samples[0].x
print(c_stars_r, flush=True)
# -------------------------------------------------------------------------------------------------------
# Warm-start and cold-start annealers
# -------------------------------------------------------------------------------------------------------
eps = 0
e_ann = solve_hm(n, Q, T, N, "warm", H, c_stars_r, eps, c_stars_r)
e_ann_es = solve_hm(n, Q, T, N, "equal", H, c_stars_r, eps, c_stars_r)
print("E:", e_ann, e_ann_es, flush = True)
print("E - offset: ", e_ann-portfolio_offset, e_ann_es-portfolio_offset, flush = True)
# -------------------------------------------------------------------------------------------------------
# Outputs saved in .json files
# -------------------------------------------------------------------------------------------------------
out = {
    "T": T,
    "N": N,
    "e_ann": e_ann-portfolio_offset,
    "e_ann_es": e_ann_es-portfolio_offset,
    "E0": E0
}
out_dir = Path("Results3")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"results_T{T}_N{N}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


