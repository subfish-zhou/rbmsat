import torch
import argparse
import time

from models.rbm import formulaRBM, formulaRBM_parallel
from data.utils import CNFFormula, unit_propagation, count_satisfied_clauses

def solve_maxsat(formula: CNFFormula, 
                 max_time=60, 
                 heuristic_interval=100, 
                 batch_size=1,
                 verbose=False,
                 F_s=-1.0,
                 num_epochs=1000,
                 lr=0.01,
                 device='cpu'):
    
    if batch_size == 1:
        rbm = formulaRBM(formula, 
                         F_s=F_s, 
                         num_epochs=num_epochs, 
                         lr=lr,
                         device=device,
                         verbose=verbose)
    else:
        rbm = formulaRBM_parallel(formula, 
                                  F_s=F_s, 
                                  num_epochs=num_epochs, 
                                  lr=lr,
                                  device=device,
                                  verbose=verbose)
    n_visible = rbm.n_visible
    
    # Initialize v to random assignment
    v = torch.bernoulli(torch.full((batch_size, n_visible), 0.5, device=device))
    
    best_v, best_num_satisfied = count_satisfied_clauses(formula, v.cpu())
    
    start_time = time.time()
    
    # Unit propagation heuristic: Initialize moving averages of ν_i
    nu_i = torch.zeros(n_visible, device=device)
    alpha = 0.9  # Decay factor for moving average
    step = 0
    while True:
        step += 1
        
        # Sample h given v
        h_sample, _ = rbm.sample_h_given_v(v)
        # Sample v given h
        v_sample, p_v_given_h = rbm.sample_v_given_h(h_sample)
        v = v_sample
        
        # Update moving averages of ν_i
        rho_i = p_v_given_h.mean(dim=0)  # size n_visible
        nu_i = alpha * nu_i + (1 - alpha) * (rho_i * (1 - rho_i))
        
        # Every heuristic_interval steps, apply heuristic
        if step % heuristic_interval == 0:
            # Rank variables by ν_i
            _, indices = torch.sort(nu_i)
            num_vars_to_unassign = n_visible // 2
            vars_to_unassign = indices[:num_vars_to_unassign]
            # Set these variables to unassigned (-1)

            assignments = []
            for i in range(batch_size):
                assignment = v[i].clone()
                assignment[vars_to_unassign] = -1  # Unassign variables
                # Apply unit propagation
                assignment = unit_propagation(formula, assignment.cpu()).to(device)
                assignments.append(assignment)
            v = torch.stack(assignments, dim=0)  # Shape: (B, N)
            # Reset moving averages
            nu_i.zero_()
        
        # Evaluate current assignment
        current_best_v, current_best_num_satisfied = count_satisfied_clauses(formula, v)
        if current_best_num_satisfied > best_num_satisfied:
            best_num_satisfied = current_best_num_satisfied
            best_v = current_best_v.clone()
        
        if best_num_satisfied == formula.num_clauses:
            print(f"Solved in {step} steps")
            break
        
        # Optional: print progress
        if step % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Step {step}, Best: {best_num_satisfied}, current: {current_best_num_satisfied}, formula size: {formula.num_clauses}, Elapsed Time: {elapsed_time:.2f}s")
        
        # Check time limit
        if time.time() - start_time >= max_time:
            print(f"Time limit reached: {max_time} seconds")
            break
            
    return best_v, best_num_satisfied


def main():
    parser = argparse.ArgumentParser(description='RbmSAT solver')
    parser.add_argument('input_file', type=str, help='Input CNF file in DIMACS format')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of parallel chains')
    parser.add_argument('--max_time', type=int, default=60, help='Maximum time in seconds')
    parser.add_argument('--heuristic_interval', type=int, default=100, help='Heuristic interval for unit propagation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on (cuda/cpu)')
    parser.add_argument('--verbose', type=bool, default=False, help='Print progress')

    # params for pre-trained RBM
    parser.add_argument('--F_s', type=float, default=-1.0, help='Free energy for pre-trained RBM')
    parser.add_argument('--num_epochs', type=int, default=10000, help='Training epochs for pre-trained RBM')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for pre-trained RBM')
    args = parser.parse_args()

    if args.verbose: print(f"Using device: {args.device}")
    
    # Read CNF file
    with open(args.input_file, 'r') as f:
        cnf_str = f.read()

    formula = CNFFormula(cnf_str)
    
    best_v, best_num_satisfied = solve_maxsat(
        formula, 
        max_time=args.max_time, 
        heuristic_interval=args.heuristic_interval, 
        batch_size=args.batch_size,
        verbose=args.verbose,
        F_s=args.F_s,
        num_epochs=args.num_epochs,
        lr=args.lr,
        device=args.device)
    print(f"Best assignment satisfies {best_num_satisfied} clauses out of {formula.num_clauses}")
    print("Best assignment:")
    print(best_v.cpu())

if __name__ == "__main__":
    main()
