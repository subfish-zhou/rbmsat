import torch
from formula import CNFFormula


def unit_propagation(formula: CNFFormula, assignment: torch.Tensor) -> torch.Tensor:
    """
    Performs unit propagation on a CNF formula given a partial assignment.
    Args:
        formula: CNFFormula object
        assignment: Tensor of size n_vars, values 0,1,-1 (-1 represents unassigned)
    Returns:
        Updated assignment after unit propagation
    """
    assignment = assignment.clone()
    changed = True
    while changed:
        changed = False
        for clause in formula.clauses:
            num_unassigned = 0
            last_unassigned_lit = None
            clause_satisfied = False
            for lit in clause:
                var = abs(lit) - 1  # 0-based index
                val = assignment[var].item()
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
                if last_unassigned_lit > 0:
                    assignment[var] = 1
                else:
                    assignment[var] = 0
                changed = True
    # For any remaining unassigned variables, set randomly
    unassigned_vars = (assignment == -1)
    assignment[unassigned_vars] = torch.bernoulli(torch.full((unassigned_vars.sum(),), 0.5, device=assignment.device))
    return assignment


def count_satisfied_clauses(formula: CNFFormula, v: torch.Tensor) -> int:
    """
    Counts the number of satisfied clauses in the CNF formula given an assignment.
    Args:
        formula: CNFFormula object
        v: Tensor of size (batch_size, n_visible), values 0 or 1
    Returns:
        Number of satisfied clauses
    """
    B = v.shape[0]  # Batch size
    num_satisfied = torch.zeros(B, device=v.device, dtype=torch.int32)

    for clause in formula.clauses:
        # Initialize a boolean tensor indicating if the clause is satisfied for each assignment
        clause_satisfied = torch.zeros(B, device=v.device, dtype=torch.bool)
        for lit in clause:
            if lit == 0:
                continue
            var = abs(lit) - 1  # 0-based index
            val = v[:, var]  # Shape: (B,)
            is_positive = lit > 0
            is_true = val.bool() if is_positive else (~val.bool())
            clause_satisfied |= is_true  # Element-wise OR across batch

            # Early exit if all assignments satisfy the clause
            if clause_satisfied.all():
                break

        # Increment count for assignments where clause is satisfied
        num_satisfied += clause_satisfied.int()

    # Find the assignment with the maximum number of satisfied clauses
    max_num_satisfied, max_idx = num_satisfied.max(dim=0)
    best_v = v[max_idx].unsqueeze(0).clone()

    return best_v, max_num_satisfied.item()