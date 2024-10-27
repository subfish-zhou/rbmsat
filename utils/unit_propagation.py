import torch
from utils.cnf_parser import CNFFormula

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