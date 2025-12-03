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
from docplex.mp.model import Model
from qiskit.primitives import StatevectorSampler
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from qiskit_optimization import QuadraticProgram
from qiskit.primitives import StatevectorEstimator
from qiskit.circuit import ParameterVector
from qiskit_optimization.converters import LinearEqualityToPenalty
from qiskit_optimization.algorithms import CplexOptimizer, MinimumEigenOptimizer
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from pprint import pprint
from typing import Optional, Union, Iterable, List, Tuple, Dict, Any
from qiskit.circuit.library import QAOAAnsatz
from qiskit_optimization.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.translators import from_ising
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
Array = Any
Tensor = Any
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parser
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Quantum Finalcial Portfolio Optimization Script")
parser.add_argument("--depth", type = int, required= True, help = "Depth of circuit")
parser.add_argument("--iter", type=int, required=True, help="Iteration number")
args = parser.parse_args()
depth = args.depth
iteration = args.iter
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Quadratic program ( docplex )
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
# Relaxation 
# -------------------------------------------------------------------------------------------------------
def relax_problem(problem) -> QuadraticProgram:
    """Change all variables to continuous."""
    relaxed_problem = copy.deepcopy(problem)
    for variable in relaxed_problem.variables:
        variable.vartype = VarType.CONTINUOUS
    return relaxed_problem
# -------------------------------------------------------------------------------------------------------
# Generate mu and sigma
# -------------------------------------------------------------------------------------------------------
def random_mu_sigma(n: int = 6, *, seed: 123, mu_range: tuple[float, float] = (0.6, 2.1), var_range: tuple[float, float] = (0.02, 0.32), corr_strength: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(mu_range[0], mu_range[1], size=n)
    A = rng.normal(size=(n, n))
    C = A @ A.T  # SPD
    d = np.sqrt(np.diag(C))
    Dinv = np.diag(1.0 / d)
    R = Dinv @ C @ Dinv      
    R = (1.0 - corr_strength) * np.eye(n) + corr_strength * R
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-8, None)
    R = (V * w) @ V.T
    variances = rng.uniform(var_range[0], var_range[1], size=n)
    S = np.diag(np.sqrt(variances))
    Sigma = S @ R @ S
    Sigma = 0.5 * (Sigma + Sigma.T)  
    return mu, Sigma
# The return vectors and covariance matrices are obtained by simulating the price of each asset following a Geometric Brownian motion for N = 250 days.
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

# QAOA Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    cost = get_expval1(params, ansatz, hamiltonian)
    return cost
# -------------------------------------------------------------------------------------------------------
# Obtains probability of obtaining a certain solution for a given circuit and cost operator.
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
def get_probs(circuit, result, pre_sol):
    optimized_circuit = circuit.assign_parameters(result.x)
    sampler = StatevectorSampler()
    pub = (optimized_circuit,)
    job = sampler.run([pub])
    counts_int = job.result()[0].data.meas.get_int_counts()
    counts_bin = job.result()[0].data.meas.get_counts()
    shots = sum(counts_int.values())
    final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
    bs = "".join("1" if float(b) > 0.5 else "0" for b in pre_sol)
    return float(final_distribution_bin.get(bs, 0.0))
# -------------------------------------------------------------------------------------------------------
# Obtains expectation value of a circuit for a given cost operator.
# -------------------------------------------------------------------------------------------------------
def get_expval1(result, candidate_circuit, hamiltonian):
    # If ansatz has a layout (because you transpiled), map the observable to it:
    estimator = StatevectorEstimator()
    job = estimator.run([(candidate_circuit, hamiltonian, result)])
    result = job.result()[0]
    cost = float(result.data.evs)
    return cost
def get_expval(result, candidate_circuit, hamiltonian, backend):
    # If ansatz has a layout (because you transpiled), map the observable to it:
    isa_hamiltonian = hamiltonian.apply_layout(candidate_circuit.layout) if getattr(candidate_circuit, "layout", None) else hamiltonian
    estimator = Estimator(mode=backend, options={"default_shots": int(2e5)})
    estimator.options.default_shots = 500000
    job = estimator.run([(candidate_circuit, isa_hamiltonian, result.x)])
    result = job.result()[0]
    cost = float(result.data.evs)
    return cost
# -------------------------------------------------------------------------------------------------------
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Porfolio Utils
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Brute force attack to find optimal solutions and costs for small problems.
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
# Returns Q matrix from portfolio problem
# -------------------------------------------------------------------------------------------------------
def QUBO_from_portfolio(cov: Array, mean: Array, q: float, B: int, t: float) -> Tensor:
    """
    convert portfolio parameters to a Q matrix
    :param cov: n-by-n covariance numpy array
    :param mean: numpy array of means
    :param q: the risk preference of investor
    :param B: budget
    :param t: penalty factor
    :return Q: n-by-n symmetric Q matrix
    """
    n = cov.shape[0]
    R = np.diag(mean)
    S = np.ones((n, n)) - 2 * B * np.diag(np.ones(n))

    Q = q * cov - R + t * S
    return Q
# -------------------------------------------------------------------------------------------------------
# Returns Pauli terms and weigths from Q matrix
# -------------------------------------------------------------------------------------------------------
def QUBO_to_Ising(Q: Tensor) -> Tuple[Tensor, List[float], float]:
    """
    Cnvert the Q matrix into a the indication of pauli terms, the corresponding weights, and the offset.
    The outputs are used to construct an Ising Hamiltonian for QAOA.

    :param Q: The n-by-n square and symmetric Q-matrix.
    :return pauli_terms: A list of 0/1 series, where each element represents a Pauli term.
    A value of 1 indicates the presence of a Pauli-Z operator, while a value of 0 indicates its absence.
    :return weights: A list of weights corresponding to each Pauli term.
    :return offset: A float representing the offset term of the Ising Hamiltonian.
    """
    n = Q.shape[0]
    offset = (np.triu(Q, 0).sum() / 2) 
    pauli_terms = []  
    weights = (-np.sum(Q, axis=1) / 2)  

    for i in range(n):
        term = np.zeros(n)
        term[i] = 1
        pauli_terms.append(term.tolist())  

    for i in range(n - 1):
        for j in range(i + 1, n):
            term = np.zeros(n)
            term[i] = 1
            term[j] = 1
            pauli_terms.append(term.tolist()) 
            weight = (Q[i][j] / 2)  
            weights = np.concatenate((weights, weight), axis=None)  

    return pauli_terms, weights, offset
def pauliZ_labels_to_vectors(op: SparsePauliOp):
    """
    Convert a SparsePauliOp into:
      - a list of 0/1 vectors (float) marking where each term has 'Z'
      - a 1D np.ndarray of coefficients (real if imag≈0)
    Notes:
      * Index 0 in each vector corresponds to qubit 0 (rightmost char in the label).
      * coeff_scale lets you rescale coefficients if you need (e.g., to match 1.5... in your example).
    """
    labels = op.paulis.to_labels()              # e.g. 'IIIIZZ'
    vecs = []
    for lab in labels:
        # lab[-1] -> qubit 0, lab[-2] -> qubit 1, ...
        v = [1.0 if c == 'Z' else 0.0 for c in lab[::-1]]
        vecs.append(v)

    coeffs = np.real_if_close(op.coeffs)        # drop tiny imaginary parts

    return vecs, coeffs
# Returns Ising SparsePauliOp from Pauli terms and weights.
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

# QAOA ansatz and solver
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# QAOA solver
# -------------------------------------------------------------------------------------------------------
def solve_hm(n, warm: bool, lukewarm: bool, cost_operator, pauli_params, pauli_weights, pre_sol, eps, depth, iteration, solution, parameters):
    circuit_w_anszt= QAOA_hm_1(n, pauli_params, pauli_weights, warm, lukewarm, pre_sol , eps, depth)
    #parms1 = create_custom_array(2*depth, -2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi)
    #parms2 = create_custom_array_even_odd(2*depth, -2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi)
    estimator = StatevectorEstimator()
    result = minimize(
        cost_func_estimator,
        parameters,
        args=(circuit_w_anszt, cost_operator, estimator),
        method="COBYLA",
        options={                 
        'tol': 1e-3,                    # stopping tolerance; try 1e-3…1e-5                # usually enough if tol is set sanely
        'disp': True
    })
    print(result)
    print("COBYLA Optimization ended", flush = True)

    probs = get_probs1(circuit_w_anszt, result, solution)
    expval = get_expval1(result.x, circuit_w_anszt, cost_operator)
    print("Probability of obtaining result string:", probs, flush=True)
    print("Expectation value of Hc after otimization:", expval, flush=True)
    return probs, expval
# -------------------------------------------------------------------------------------------------------
# QAOA ansatz builder
# -------------------------------------------------------------------------------------------------------
def QAOA_hm_1(n, pauli_params, pauli_weights, warm: bool, lukewarm : bool, pre_sol , eps, depth):
    c_stars=[]
    print(f"depth is {depth}", flush=True)
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
    init_qc = init(n, warm, thetas)
    circuit = QuantumCircuit(n, name="QAOA")
    circuit.append(init_qc, range(n))
    for j in range(depth):
        qc_p_j = cost(n, pauli_params, pauli_weights, gammas[j])  

        if lukewarm:                                               
            mixer_qc_j = mixer(n, False, thetas, betas[j])
        else:
            mixer_qc_j = mixer(n, warm, thetas, betas[j])
        circuit.append(qc_p_j, range(n))
        circuit.append(mixer_qc_j, range(n))
    return circuit
# -------------------------------------------------------------------------------------------------------
# Create cost hamiltonian circuit for QAOA
# -------------------------------------------------------------------------------------------------------
def cost(n,pauli_terms: List[List[float]],weights: List[float],gamma):
    """
    Append the QAOA cost unitary U_C(gamma) = exp(-i * gamma * H_C)
    for H_C = sum_k w_k * Z^{(term_k)} (offset ignored as global phase).

    pauli_terms: list of 0/1 masks (floats ok), len = m, each of length n (num qubits)
    weights:     list/array of length m
    gamma:       float or qiskit Parameter
    """
    qc_p = QuantumCircuit(n)
    w = np.asarray(weights, dtype=float)
    for wk, term in zip(w, pauli_terms):
        idx = [i for i, b in enumerate(term) if b > 0.5]
        if len(idx) == 1:
            q = idx[0]
            # RZ(θ) = exp(-i θ Z / 2)  => θ = 2 * gamma * w
            qc_p.rz(wk * gamma, q)
        elif len(idx) == 2:
            i, j = idx
            # RZZ(θ) = exp(-i θ ZZ / 2) => θ = 2 * gamma * w
            qc_p.rzz(wk * gamma, i, j)
    return qc_p
# -------------------------------------------------------------------------------------------------------
# Create mixer hamiltonian circuit for QAOA
# -------------------------------------------------------------------------------------------------------
def mixer(n, warm: bool, thetas, beta):
    ws_mixer = QuantumCircuit(n)
    if warm:
        for idx, theta in enumerate(thetas):
            ws_mixer.ry(-theta, idx)
            ws_mixer.rz(-2 * beta, idx)
            ws_mixer.ry(theta, idx)
    else:
        for i in range(n):
            ws_mixer.rx(2 * beta, i)
    return ws_mixer
# -------------------------------------------------------------------------------------------------------
# Create initial state circuit for QAOA
# -------------------------------------------------------------------------------------------------------
def init(n, warm: bool, thetas):
    init_qc = QuantumCircuit(n)
    if warm:
        for idx, theta in enumerate(thetas):
            init_qc.ry(theta, idx)
    else:
        for i in range(n):
            init_qc.h(i)
    return init_qc
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Energy of ground state and given bitstring
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def energy_from_terms(bits: List[int], pauli_terms: List[List[float]], weights: List[float], offset: float = 0.0):
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
# -------------------------------------------------------------------------------------------------------
def ground_energy_from_terms(pauli_terms: List[List[int]], weights: List[float], offset: float = 0.0) -> Tuple[float, List[int]]:
    """
    Compute E0 and one minimizing bitstring for an Ising-type Hamiltonian:
        H = sum_k w_k * Z^{(term_k)} + offset * I
    pauli_terms[k] is a 0/1 mask over qubits; weights[k] is its coefficient.
    Returns (E0, bitstring) with bitstring in {0,1}^n (0->|0>, 1->|1>).

    Conventions:
      - Qubit indexing follows term indices: term[0] acts on qubit 0, etc.
      - Mapping to spins: z_i = (-1)^{b_i} = 1 - 2*b_i.
    """
    pauli_terms = [list(t) for t in pauli_terms]
    weights = np.asarray(weights, dtype=float)
    n = len(pauli_terms[0])
    m = len(pauli_terms)

    num = 1 << n
    a = np.arange(num, dtype=np.uint64)
    # bits[:, i] is bit of qubit i (LSB = qubit 0)
    shifts = np.arange(n, dtype=np.uint64)          
    bits = ((a[:, None] >> shifts) & np.uint64(1)).astype(np.int8)
    z = 1 - 2 * bits  # 0->+1, 1->-1

    term_indices = [np.flatnonzero(t) for t in pauli_terms]

    E = np.full(num, offset, dtype=float)
    for w, idx in zip(weights, term_indices):
        if len(idx) == 1:
            E += w * z[:, idx[0]]
        elif len(idx) == 2:
            i, j = int(idx[0]), int(idx[1])
            E += w * (z[:, i] * z[:, j])

    # Ground energy and a minimizer
    k_min = int(np.argmin(E))
    E0 = float(E[k_min])
    ground_bits = bits[k_min].tolist()  # [b0, b1, ..., b_{n-1}]
    return E0, ground_bits
# Computes minimum energy based on the eigenvalues of the cost operator.
# -------------------------------------------------------------------------------------------------------
def ground_energy_from_qp(qp, include_offset=True):
    # Ensure QUBO form first (safe even if already QUBO)
    qp_qubo = QuadraticProgramToQubo().convert(qp)
    op, offset = qp_qubo.to_ising()                   # op: SparsePauliOp (diagonal Z-only)

    mes = NumPyMinimumEigensolver()                   # exact eigensolver
    res = mes.compute_minimum_eigenvalue(op)

    E0 = float(np.real(res.eigenvalue))               # ground energy of 'op' only
    return E0, E0 + (float(offset) if include_offset else 0.0)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parameter initialization
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create array of random numbers that repeat themselves in even and odd positions, respectively.
# -------------------------------------------------------------------------------------------------------
def create_custom_array_even_odd(length: int, lower1: float, upper1: float, lower2: float, upper2: float):

    even_val = np.random.uniform(lower1, upper1)
    odd_val  = np.random.uniform(lower2, upper2)
    arr = np.empty(length, dtype=float)
    arr[0::2] = even_val  # even indices
    arr[1::2] = odd_val   # odd indices

    return arr
# -------------------------------------------------------------------------------------------------------
# Creates array of random numbers that repeat for each side.
# -------------------------------------------------------------------------------------------------------
def create_custom_array(length: int, lower1: float, upper1: float, lower2: float, upper2: float) -> np.ndarray:
    """
    Creates an array of a given length where the first half has identical random values between
    lower1 and upper1, and the second half has identical random values between lower2 and upper2.
    """
    if length % 2 != 0:
        raise ValueError("Length must be an even number to split into two halves.")
    first_half_value = np.random.uniform(lower1, upper1)
    second_half_value = np.random.uniform(lower2, upper2)
    array = np.concatenate((np.full(length // 2, first_half_value),np.full(length // 2, second_half_value)))

    return array
# -------------------------------------------------------------------------------------------------------
# Creates array of random numbers.
# -------------------------------------------------------------------------------------------------------
def rand_vec(i: int, seed) -> np.ndarray:
    """[-2π, 2π]"""
    rng = np.random.default_rng(seed)
    return rng.uniform(-2*np.pi, 2*np.pi, size=2*i)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Financial Portfolio parameters
# -------------------------------------------------------------------------------------------------------
n = 6
seed = 123
mu, sigma = simulate_portfolio_gbm(n, 250, mu_range = (-0.05, 0.05), sigma_range = (-0.20, 0.20),  seed = seed)
B = 3                                                                                                             # Number of selected assets
l = 3                                                                                                             # Lambda for penalty term
q = 2                                                                                                             # Risk-return trade-off


'''
mdl = Model("portfolio_optimization")
x = mdl.binary_var_list(range(n), name="x")
objective = mdl.sum([mu[i] * x[i] for i in range(n)])
objective -= q * mdl.sum([sigma[i, j] * x[i] * x[j] for i in range(n) for j in range(n)])
mdl.maximize(objective)
mdl.add_constraint(mdl.sum(x[i] for i in range(n)) == B)
qp = from_docplex_mp(mdl)
linear2penalty = LinearEqualityToPenalty(penalty=l)
qp2 = linear2penalty.convert(qp)'''


# -------------------------------------------------------------------------------------------------------
# Creation of problem and cost operator
# -------------------------------------------------------------------------------------------------------

Q = QUBO_from_portfolio(sigma, mu, q, B, l)
portfolio_pauli_terms, portfolio_weights, portfolio_offset = QUBO_to_Ising(Q)
print("Ground state energy and string (brute force)", ground_energy_from_terms(portfolio_pauli_terms, portfolio_weights, portfolio_offset), flush = True)
print(portfolio_pauli_terms, portfolio_weights, portfolio_offset, flush = True)
H = ising_from_terms_qiskit(portfolio_pauli_terms, portfolio_weights, portfolio_offset)

qp2 = create_problem1(mu, sigma, B, q, l)

'''cost_operator, offset = qp2.to_ising()
vecs, weights = pauliZ_labels_to_vectors(cost_operator)'''
#qp = from_ising(H, offset=portfolio_offset)

solver_q = NumPyMinimumEigensolver()
res_q = solver_q.compute_minimum_eigenvalue(H)
E0_q = res_q.eigenvalue.real              # ground-state energy (float)
eigenstate = res_q.eigenstate
print( "Qiskit E0:", E0_q, flush = True)
print( "Qiskit Eigenstate:", eigenstate, flush = True)
# -------------------------------------------------------------------------------------------------------
# Classical solution using CPLEX
# -------------------------------------------------------------------------------------------------------
#qp = create_problem1(mu, sigma, B, q, l)
sol= CplexOptimizer().solve(qp2)
c_stars = sol.samples[0].x
print("Classical solution CPLEX", c_stars, flush=True)
# -------------------------------------------------------------------------------------------------------
# Ground state energy 
# -------------------------------------------------------------------------------------------------------
E0 = energy_from_terms(c_stars, portfolio_pauli_terms, portfolio_weights)
print("Energy:", E0)
print("Energy + offset:", E0 + portfolio_offset)
# -------------------------------------------------------------------------------------------------------
# Relax problem and solve classically using CPLEX
# -------------------------------------------------------------------------------------------------------
qp_r = relax_problem(qp2)
sol_r= CplexOptimizer().solve(qp_r)
c_stars_r = sol_r.samples[0].x
print("Classical solution relaxed CPLEX",c_stars_r, flush=True)
# -------------------------------------------------------------------------------------------------------
# Warm-start QAOA, Cold-start QAOA, Warm-start with normal mixer QAOA 
# -------------------------------------------------------------------------------------------------------
rng = np.random.default_rng(seed+15)
eps = 0
params = create_custom_array(2*depth, -2*np.pi, 2*np.pi, -2*np.pi, 2*np.pi)
pw, ew = solve_hm(n, True, False, H, portfolio_pauli_terms, portfolio_weights, c_stars_r, 0.01, depth, iteration, c_stars, params)
pc, ec = solve_hm(n, False, False, H, portfolio_pauli_terms, portfolio_weights, c_stars_r, 0.01, depth, iteration, c_stars, params)
plw, elw = solve_hm(n, True, True, H, portfolio_pauli_terms, portfolio_weights, c_stars_r, 0.01, depth, iteration, c_stars, params)
print("Solved", flush=True)
# -------------------------------------------------------------------------------------------------------
# Outputs saved in .json files
# -------------------------------------------------------------------------------------------------------
out = {
    "depth": depth,
    "iteration": iteration,
    "temp_pw": pw,
    "temp_pc": pc,
    "temp_plw": plw,
    "temp_ew": ew-portfolio_offset,
    "temp_ec": ec-portfolio_offset,
    "temp_elw": elw-portfolio_offset,
    "E0": E0,
}
out_dir = Path("/mnt/netapp1/Store_CESGA/home/cesga/jsouto/WS_GitHub/ContinuousStart/Results1")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"results_d{depth}_i{iteration}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)   
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------