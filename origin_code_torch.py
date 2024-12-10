import time
import torch
import torch.nn.functional as F
import logging
from copy import copy

from utils import CNFFormula
from models.rbm import clauseRBM
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)

def send_for_unit_propagation(assignments: torch.Tensor, variances: torch.Tensor) -> torch.Tensor:
    """
    Prepare a partial assignment for unit propagation by 'freezing' 
    the half of the variables with the highest variance and setting 
    the other half to -1 (unassigned).

    Parameters
    ----------
    assignments : torch.Tensor
        Current variable assignments (batch_size, num_vars), with values in {0, 1}.
    variances : torch.Tensor
        Estimated variances for each variable (batch_size, num_vars).

    Returns
    -------
    torch.Tensor
        Modified assignments where low-variance variables are set to -1.
    """
    batch_size, num_vars = assignments.shape
    # Determine how many variables to keep fixed
    half = num_vars // 2
    modified = assignments.clone()

    for b in range(batch_size):
        variance = variances[b]
        # Sort by descending variance
        sorted_indices = torch.argsort(variance, descending=True)
        # Set lower half to -1 (unassigned)
        modified[b, sorted_indices[half:]] = -1

    return modified

def fetch_unit_prop_result(formula: CNFFormula, partial_assignments: torch.Tensor) -> torch.Tensor:
    """
    Apply unit propagation to the given partial assignments using the CNFFormula directly.

    Parameters
    ----------
    formula : CNFFormula
        The formula object containing clauses as a list of lists of integers.
    partial_assignments : torch.Tensor
        A (batch_size, num_vars) tensor representing partially unassigned assignments.
        -1 indicates an unassigned variable, and 0/1 are assigned values.

    Returns
    -------
    torch.Tensor
        A (batch_size, num_vars) tensor with updated assignments after unit propagation.
    """
    results = []
    for assignment in partial_assignments.cpu():
        new_assignment = assignment.clone().detach()
        changed = True
        while changed:
            changed = False
            for clause in formula.clauses:
                num_unassigned = 0
                last_unassigned_lit = None
                clause_satisfied = False
                for lit in clause:
                    var = abs(lit) - 1  # 0-based index
                    val = new_assignment[var].item()
                    if val == -1:
                        num_unassigned += 1
                        last_unassigned_lit = lit
                    else:
                        is_true = (val == 1 and lit > 0) or (val == 0 and lit < 0)
                        if is_true:
                            clause_satisfied = True
                            break
                if not clause_satisfied and num_unassigned == 1:
                    # Unit clause, assign the last unassigned variable to satisfy the clause
                    var = abs(last_unassigned_lit) - 1
                    new_assignment[var] = 1 if last_unassigned_lit > 0 else 0
                    changed = True
        results.append(new_assignment)
    return torch.stack(results).to(partial_assignments.device)

def gather_and_count(assignments, clause_var_matrix, polarity):
    """
    Compute the clause satisfaction count for given assignments.
    """
    c = torch.einsum('bv,ckv->bck', assignments, clause_var_matrix) 
    return c, (((polarity + c) == 2) + ((polarity + c) == -1)).any(dim=-1).sum(dim=-1)


def time_remaining(start_time, time_limit):
    return (time.time() - start_time) <= time_limit


def construct_Q(clauses, num_var, device):
    Q = torch.zeros(clauses.shape[0], clauses.shape[1], num_var, device=device)
    for clause_idx, clause in enumerate(clauses):
        for lit_idx, lit in enumerate(clause):
            if lit != 0:
                Q[clause_idx, lit_idx, abs(lit)-1] = 1.0
    return Q


def rbmsat(formula, W_c, b_c, batch_size, seed, time_limit, heuristic_inteval=100, heuristic_delay=1, alpha=0.1):
    """
    RBM-based SAT solver main loop.

    Parameters
    ----------
    formula : CNFFormula
        The CNF formula object.
    W_c : torch.Tensor
        Clause RBM weight matrix.
    b_c : torch.Tensor
        Clause RBM bias vector.
    batch_size : int
        Batch size for parallel runs.
    seed : int
        Random seed.
    time_limit : float
        Maximum allowed time in seconds.
    upp : int
        Unit propagation period.
    upw : int
        Weight for unit propagation steps.
    alpha : float
        Smoothing parameter for variance updates.

    Returns
    -------
    s_max_list : list
        A list of tuples (step, max_satisfied_clauses).
    best_assignment : torch.Tensor
        The best assignment found.
    """
    formula_padding_copy = copy(formula)
    formula_padding_copy.padding()
    clauses = torch.tensor(formula_padding_copy.clauses, device=device)
    num_vars = formula.num_vars

    torch.manual_seed(seed)
    assignments = torch.bernoulli(torch.ones(batch_size, num_vars, device=W_c.device)*0.5)
    variances = 0.25 * torch.ones_like(assignments)  # initial variance guess

    polarity = torch.sign(clauses)
    clause_var_matrix = construct_Q(clauses, num_vars, device)
    W = torch.einsum('ck,kh->ckh', polarity, W_c)
    b = b_c.repeat([clauses.shape[0], 1]) + torch.mm((1 - polarity) / 2, W_c)

    step, d = 1, -1
    s_max_list = []
    best_assignment = torch.ones(num_vars, device=W_c.device)
    s_max = 0
    start_time = time.time()

    # Using these variables to store partial states for unit propagation
    last_partial = assignments.clone()
    with torch.no_grad():
        while time_remaining(start_time, time_limit):

            if heuristic_inteval > 0 and d == 0:
                # print("use heuristic")
                # Fetch updated assignments from unit propagation
                up_assignments = fetch_unit_prop_result(formula, last_partial)
                combined = torch.vstack([assignments, up_assignments])
                _, s_u = gather_and_count(combined, clause_var_matrix, polarity)
                ranks = torch.argsort(s_u)
                assignments = combined[ranks[-batch_size:]]
                variances = torch.vstack([variances, variances])[ranks[-batch_size:]]

            if heuristic_inteval > 0 and step % heuristic_inteval == 0:
                d = heuristic_delay
                # Update partial assignments for unit propagation
                last_partial = send_for_unit_propagation(assignments, variances)

            c, s = gather_and_count(assignments, clause_var_matrix, polarity)
            s_max_step, idx_max = s.max(dim=0)
            if s_max_step.item() > s_max:
                s_max = s_max_step.item()
                best_assignment = assignments[idx_max].clone()
                s_max_list.append((step, s_max))
                logging.info(f"New s_max found {s_max} at step {step}")

            h_logits = b + torch.einsum('bck,ckh->bch', c, W)
            h = torch.bernoulli(torch.sigmoid(h_logits))
            ro = torch.sigmoid(torch.einsum('bch,ckh,ckv->bv', h, W, clause_var_matrix))
            variances = (1 - alpha) * variances + alpha * ro * (1 - ro)
            assignments = torch.bernoulli(ro)

            step, d = step + 1, max(d - 1, -1)
    
    s_max_list.append((step, s_max_list[-1][-1]))

    return s_max_list, best_assignment


if __name__ == "__main__":
    # Example usage
    with open("wcnfdata/brock200_3.clq.wcnf", 'r') as f:
        cnf_str = f.read()

    formula = CNFFormula(cnf_str)
    rbm = clauseRBM(formula.max_clause_length, device=device)

    batch_size = 1024
    seed = 42
    timeout = 3000
    alpha, heuristic_inteval, heuristic_delay = 0.1, 5000, 1

    s_max_list, best_assignment = rbmsat(formula, rbm.W, rbm.b, batch_size, seed, timeout, heuristic_inteval, heuristic_delay, alpha)
    logging.info(f"Best assignment found: {best_assignment}")

    # Optional: plot results
    if s_max_list:
        x_ax, y_ax = zip(*s_max_list)
        plt.step(x_ax, y_ax)
        plt.xlabel('Step')
        plt.ylabel('Max Satisfied Clauses')
        plt.show()
