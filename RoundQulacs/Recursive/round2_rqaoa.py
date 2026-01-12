# Imports
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import copy
import numpy as np
import csv
import json
import argparse
import random 
import cvxpy as cvx
import os
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from qulacs import QuantumState, QuantumCircuit, QuantumCircuitSimulator
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
from collections import defaultdict
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parser
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Recursive Quantum Approximate Optimization Algorithm for MAXCUT (Qulacs)")
parser.add_argument("--n", type = int, required= True, help = "Number of nodes in graph")
parser.add_argument("--flag", type = str, required= True, help = "Type of graph")
parser.add_argument("--iter", type = int, required= True, help = "Number of graph")
parser.add_argument("--case", type=str, required=True, choices=["CSQAOA", "WSQAOA", "GW", "RGW"],
                help='Which method to run: "CSQAOA", "WSQAOA", "GW", "RGW"')
args = parser.parse_args()
n_ = args.n
flag = args.flag.lower() == "true"
iteration = args.iter
case = args.case.upper()
print(n_, flush  = True)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Graph generation 
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def gen_maxcut(n: int, fc: bool, w_min: float, w_max: float, seed=None) -> np.ndarray:

    py_rng = random.Random(seed)
    W = np.zeros((n, n), dtype=float)

    if fc:
        pw = np.linspace(w_min, w_max, num=21)  
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

# RQAOA solver
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class RQAOA():

    def __init__(self, maxcut, n_init, warm: bool, depth=1, n_cuts=5, n_limit=None, eps=0.25):
        self.maxcut = maxcut
        self.n_init = n_init
        self.depth = depth
        self.eps = eps
        self.n_cuts = n_cuts
        self.n_limit = n_init // 2 if n_limit is None else n_limit
        self.warm = warm

    def gw_sdp(self, graph, num_cuts, seed: int = 0):
        """
        Run the Goemans–Williamson SDP relaxation for Max-Cut using CVXPY,
        then perform random hyperplane rounding to generate binary cuts.

        Args:
            num_cuts: number of random cuts to generate.
            seed:     RNG seed for reproducibility.

        Returns:
            cuts: ndarray of shape (num_cuts, n) with entries in {0,1}, each row is one cut.
        """

        W = graph
        n = W.shape[0]

        rng = np.random.default_rng(seed)

        # Solve SDP: maximize sum_{i,j} W_ij (1 - X_ij) ---
        X = cvx.Variable((n, n), PSD=True)
        constraints = [X[i, i] == 1 for i in range(n)]

        ones = np.ones((n, n))
        objective_expr = cvx.sum(cvx.multiply(W, (ones - X)))
        problem = cvx.Problem(cvx.Maximize(objective_expr), constraints)
        problem.solve()

        chi = X.value  # SDP solution 

        # Make chi numerically PSD and well-conditioned for Cholesky 

        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(chi)
        eigvals_clipped = np.clip(eigvals, a_min=1e-12, a_max=None)
        chi_psd = (eigvecs * eigvals_clipped) @ eigvecs.T

        # Ensure diagonal ~ 1 
        d = np.sqrt(np.diag(chi_psd))
        d[d == 0.0] = 1.0
        D_inv = np.diag(1.0 / d)
        chi_psd = D_inv @ chi_psd @ D_inv

        # Small jitter in case Cholesky is still unhappy
        try:
            L = np.linalg.cholesky(chi_psd).T
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(chi_psd + 1e-8 * np.eye(n)).T

        # Random hyperplane rounding

        # r ~ N(0, I), shape (num_cuts, n)
        r = rng.normal(size=(num_cuts, n))
        y = r @ L 

        # Cut bits: sign > 0 => set 1, else 0
        cuts = (y > 0).astype(int)  
        return cuts

    def build_cost_hamiltonian(self, n, graph):
        terms: List[List[int]] = []
        weights: List[float] = []
        Q = graph
        offset = np.triu(Q, 0).sum() / 2.0
        reverse_qubit_order = True
        # Obtain Pauli terms and weigths    
        for i in range(n - 1):
            for j in range(i + 1, n):
                if Q[i, j] != 0: 
                    term = np.zeros(n, dtype=int)
                    term[i] = 1
                    term[j] = 1
                    terms.append(term.tolist())  
                    weight = Q[i, j] / 2.0
                    weights.append(float(weight))
        
        # Obtain Qulacs observable
        T = np.asarray(terms, dtype=float)
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
            masks.append(mask)  

        observable: Optional[Observable] = None

        observable = Observable(n)
        for mask, coeff in zip(masks, w):
            pauli_ops = []
            for q, active in enumerate(mask):
                if active:
                    pauli_ops.append(f"Z {q}")
            if pauli_ops:
                op_str = " ".join(pauli_ops)
                observable.add_operator(float(coeff), op_str)

        return terms, weights, observable

    def build_qaoa_ansatz(self, cut, params, terms, weights):
        # -------------------------------------------------------------------------------------------------------
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
        c_stars=[]
        for var in cut:
            if var <= self.eps:
                c_stars.append(self.eps)     
            elif var >= 1-self.eps:    
                c_stars.append(1-self.eps)
            else:
                c_stars.append(var)
        thetas = [2 * np.arcsin(np.sqrt(c_star)) for c_star in c_stars]
        gammas = params[self.depth:] 
        betas  = params[:self.depth]  
        n = len(cut)
        circuit = QuantumCircuit(n)
        init_circ(circuit, n, self.warm, thetas)

        for j in range(self.depth):
            cost_circ(circuit, n, terms, weights, gammas[j])
            mixer_circ(circuit, n, self.warm, thetas, betas[j])                 
        return circuit

    def cut_value(self, x, W):
            """
            Compute value of a cut x for MaxCut matrix W.
            x is a binary vector in {0,1}^n.
            """
            cut_mat = np.outer(x, (1 - x))
            return float(np.sum(W * cut_mat))

    def get_cuts(self, graph, seed: int = 0, sort: bool = True, unique: bool = True):
        """
        Generate a specific number of cuts from the GW SDP solution.
        """

        W = graph
        cuts = self.gw_sdp(graph, num_cuts=self.n_cuts, seed=seed)
        values = np.array([self.cut_value(cut, W) for cut in cuts], dtype=float)

        if sort:
            order = np.argsort(-values)
            cuts = cuts[order]
            values = values[order]

        cuts = cuts[:self.n_cuts]
        values = values[:self.n_cuts]
        return cuts, values
    
    def grid_search(self, observable, cut, terms, weights):
        # -------------------------------------------------------------------------------------------------------
        def E(a, b, beta):
            return a*np.sin(2*beta) + b*np.sin(4*beta)
        # -------------------------------------------------------------------------------------------------------
        def ab(E_beta1: float, E_beta2: float):
            a = -E_beta1
            b = -E_beta1 * (np.sqrt(2)/2) - E_beta2
            return a, b
        # -------------------------------------------------------------------------------------------------------    
        def beta(a: float, b: float, domain=(3*np.pi/8, 3*np.pi/4)):
            """
            Minimize the surrogate f(β) = a sin(2β) + b sin(4β) over [β_min, β_max] 
            """
            beta_min, beta_max = domain
            x0 = np.array([(beta_min + beta_max)/2.0], dtype=float)

            def model(beta_arr):
                beta = beta_arr[0]
                return E(a, b, beta)  # surrogate energy

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
        # -------------------------------------------------------------------------------------------------------
        gamma_grid = np.linspace(-2*np.pi, 2*np.pi, 25)                         
        beta_probes = np.array([3*np.pi/4, 3*np.pi/8])    
        beta_star_per_gamma1 = []
        E_star_per_gamma1 = []
        for g in gamma_grid:
            e1 = self.cost([beta_probes[0], g], observable, cut, terms, weights)
            e2 = self.cost([beta_probes[1], g], observable, cut, terms, weights)
            a, b = ab(e1, e2)
            beta_star1, E_star_model1 = beta(a, b, domain=(3*np.pi/8, 3*np.pi/4))
            E_star_true1 = self.cost([beta_star1, g], observable, cut, terms, weights)
            beta_star_per_gamma1.append(beta_star1)
            E_star_per_gamma1.append(E_star_true1)
        beta_star_per_gamma1 = np.array(beta_star_per_gamma1)
        E_star_per_gamma1 = np.array(E_star_per_gamma1)
        best_idx1    = int(np.argmin(E_star_per_gamma1))   
        gamma_star1  = float(gamma_grid[best_idx1])
        beta_star1   = float(beta_star_per_gamma1[best_idx1])
        E_star_best1 = float(E_star_per_gamma1[best_idx1])
        return [gamma_star1, beta_star1]

    def cost(self, params, observable, cut, terms, weights):
        circuit = self.build_qaoa_ansatz(cut, params, terms, weights)
        state = QuantumState(len(cut))
        circuit.update_quantum_state(state)
        return observable.get_expectation_value(state)
    
    def pwf_solver(self, params, cut, terms, weights):
        def state_to_distribution_bin(state: QuantumState, tol: float = 1e-12):
            """
            Convert a Qulacs QuantumState into a dict {bitstring: probability}.
            """
            n = state.get_qubit_count()
            vec = state.get_vector()  
            dist = {}
            for idx, amp in enumerate(vec):
                p = abs(amp)**2
                if p < tol:
                    continue 
                bitstring = format(idx, f"0{n}b")
                dist[bitstring] = float(p)
            norm = sum(dist.values())
            if norm > 0:
                for k in dist:
                    dist[k] /= norm
            return dist
        circuit = self.build_qaoa_ansatz(cut, params, terms, weights)
        state = QuantumState(len(cut))
        sim = QuantumCircuitSimulator(circuit, state)
        sim.simulate()
        return state_to_distribution_bin(state)

    def solve_qaoa(self, cut, observable, terms, weights):
        params = self.grid_search(observable, cut, terms, weights)
        circuit = self.build_qaoa_ansatz(cut, params, terms, weights)
        result = minimize(self.cost,
            params,
            args=(observable, cut, terms, weights),
            method="COBYLA",
            options={
            'rhobeg': 0.4,                  # ~10% of variable scale; tune 0.05–0.5
            'tol': 1e-4,                    # stopping tolerance; try 1e-3…1e-5 
            'maxiter': 100000,                         
        })
        #expval = cost(result.x, cut, terms, weights)
        pwf = self.pwf_solver(result.x, cut, terms, weights)
        return pwf

    def build_corr_matrix(self, pwfs):
        N = len(pwfs)
        acc = defaultdict(float)
        for d in pwfs:
            for bitstring, p in d.items():
                acc[bitstring] += p
        avg = {bit: acc[bit] / N for bit in acc}
        bitstrings = list(avg.keys())
        probs = np.array([avg[b] for b in bitstrings])
        probs /= probs.sum()  
        n = len(bitstrings[0])  
        M = np.zeros((n, n))
        for b, p in zip(bitstrings, probs):
            z = np.array([1 if bit == '0' else -1 for bit in b])
            M += p * np.outer(z, z)
        return M

    def reduce_graph(self, G0, M, nodes):
        """
        One RQAOA reduction step, working directly on a weight matrix W instead of a networkx graph.

        Args:
            W: (n, n) weight matrix of the current graph.
            M: (n, n) matrix used to pick the pair (i, j) and alpha (e.g., correlation matrix).
            nodes: list of original node labels corresponding to indices 0..n-1 of W and M.

        Returns:
            W2: reduced weight matrix after folding node i into j and removing i.
            nodes2: updated list of node labels.
            (i_node, j_node, alpha): the contraction info in terms of original node labels.
        """
        assert G0.shape == M.shape, "W and M must have the same shape"
        n = G0.shape[0]
        assert n == len(nodes), "len(nodes) must match matrix size"

        # zero-out diagonal of M to ignore self-terms
        M_off = M.copy()
        np.fill_diagonal(M_off, 0.0)

        # pick pair (i_idx, j_idx) = argmax |M_ij|
        i_idx, j_idx = np.unravel_index(np.argmax(np.abs(M_off)), M_off.shape)
        i_node, j_node = nodes[i_idx], nodes[j_idx]
        alpha = 1 if M[i_idx, j_idx] >= 0 else -1

        # copy W so we can modify
        W2 = G0.copy()

        # fold row/column i into row/column j: w_{jk} += alpha * w_{ik}
        for k_idx in range(n):
            if k_idx in (i_idx, j_idx):
                continue

            w_ik = W2[i_idx, k_idx]
            if w_ik == 0.0:
                continue

            w_jk = W2[j_idx, k_idx]
            new_w = w_jk + alpha * w_ik

            # update symmetric entries
            W2[j_idx, k_idx] = new_w
            W2[k_idx, j_idx] = new_w

        # now remove node i_idx from the matrix:
        # delete its row and column
        W2 = np.delete(W2, i_idx, axis=0)
        W2 = np.delete(W2, i_idx, axis=1)

        # update nodes list: drop the i_idx-th entry
        nodes2 = [u for k, u in enumerate(nodes) if k != i_idx]

        return W2, nodes2, (i_node, j_node, alpha)

    def rebuild_solution(self, graph, remaining_nodes):
        all_nodes = list(range(self.n_init))
        small_cut = self.gw_sdp(graph, num_cuts=1, seed= 0)
        x_small = np.asarray(small_cut).ravel()
        unique_vals = set(np.unique(x_small).tolist())
        if unique_vals.issubset({0, 1}):
            z_small = 1 - 2 * x_small.astype(int)   # 0->+1, 1->-1
        elif unique_vals.issubset({-1, 1}):
            z_small = x_small.astype(int)
        else:
            raise ValueError("x_small must be bits {0,1} or spins {-1,+1}")
        z_full = {node: int(z_small[i]) for i, node in enumerate(remaining_nodes)}
        for i_node, j_node, alpha in reversed(self.elims):
            if j_node not in z_full:
                raise KeyError(f"Back-substitution failed: spin for j_node {j_node} not assigned yet.")
            z_full[i_node] = int(alpha * z_full[j_node])
        x_full_bits = np.array([(1 - z_full[v]) // 2 for v in all_nodes], dtype=int)
        return x_full_bits

    def recursive(self):
        G0 = self.maxcut
        print("Start of recursive QAOA.", flush = True)
        n_current = self.n_init
        nodes0 = list(range(n_current))
        self.terms, self.weights, self.observable = self.build_cost_hamiltonian(n_current, G0)
        self.elims = []
        while n_current > self.n_limit:
            print(f"Recursive QAOA with {n_current} variables.", flush = True)
            terms, weights, observable = self.build_cost_hamiltonian(n_current, G0)
            if self.warm:
                cuts, _ = self.get_cuts(G0)
            else: 
                cuts = [list(range(n_current))]
                print(len(cuts), flush = True)
            print("Cuts generated.", flush = True)
            pwfs = []
            for i in range(len(cuts)):
                pwf = self.solve_qaoa(cuts[i], observable, terms, weights)
                pwfs.append(pwf)
            print("Probability distributions obtained.", flush = True)
            M = self.build_corr_matrix(pwfs)
            print("Correlation matrix obtained.", flush = True)
            G0, nodes0, steps0 = self.reduce_graph(G0, M, nodes0)
            print("Graph reduced.", flush = True)
            self.elims.append(steps0)
            n_current = n_current - 1
        print("Recursive process ended.", flush = True)
        return self.rebuild_solution(G0, nodes0)
    
    def gw_recursive(self):
        G0 = self.maxcut
        print("Start of recursive GW.", flush = True)
        n_current = self.n_init
        nodes0 = list(range(n_current))
        self.elims = []
        while n_current > self.n_limit:
            print(f"Recursive GW with {n_current} variables.", flush = True)
            cuts, _ = self.get_cuts(G0)
            print("Cuts generated.", flush = True)
            M = self.build_corr_matrix([ { ''.join(map(str, cut.tolist())): 1.0 } for cut in cuts ])
            print("Correlation matrix obtained.", flush = True)
            G0, nodes0, steps0 = self.reduce_graph(G0, M, nodes0)
            print("Graph reduced.", flush = True)
            self.elims.append(steps0)
            n_current = n_current - 1
        print("Recursive process ended.", flush = True)
        return self.rebuild_solution(G0, nodes0)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Data storage functions
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _to_jsonable_solution(sol):
    if isinstance(sol, np.ndarray):
        return sol.astype(int).tolist()
    if isinstance(sol, (list, tuple)):
        return [int(x) for x in sol]
    return sol 

def update_results_json(path: Path, n_: int, iteration: int, case: str, sol, cut_value: float):
    data = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    data.setdefault("n", int(n_))
    data.setdefault("iter", int(iteration))
    data[case] = {
        "solution": _to_jsonable_solution(sol),
        "cut_value": float(cut_value),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Script  
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    
    G = gen_maxcut(n_, True, -10, 10, iteration)

    if case == "WSQAOA":
        rq = RQAOA(G, n_, flag, eps=0.25)
        sol = rq.recursive()

    elif case == "CSQAOA":
        rq = RQAOA(G, n_, False)
        sol = rq.recursive()

    elif case == "GW":
        rq = RQAOA(G, n_, flag, n_cuts=1)
        sol = rq.get_cuts(G, seed=seed)[0][0]  

    elif case == "RGW":
        rq = RQAOA(G, n_, flag, n_cuts=5)
        sol = rq.gw_recursive()
    else:
        raise ValueError(f"Unknown case: {case}")

    print(sol, flush=True)

    cut_value = rq.cut_value(sol, G)
    print(f"Cut value {cut_value} for case {case}.",  flush=True)

    # Write results in .json file
    out_path = Path(f"Results2/results_{n_}_{iteration}.json")
    update_results_json(out_path, n_, iteration, case, sol, cut_value)
    print(f"Saved results to {out_path}", flush=True)

if __name__ == "__main__":
    main()
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------