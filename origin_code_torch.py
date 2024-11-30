from utils import CNFFormula
from models.rbm import clauseRBM
import torch
import time
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("wcnfdata/brock200_3.clq.wcnf", 'r') as f:
    cnf_str = f.read()

formula = CNFFormula(cnf_str)
formula.padding()
rbm = clauseRBM(formula.max_clause_length)

'''
# Global variable to store the result of unit propagation
# unit_prop_result = None
# last_v = None  # To store the last variable assignments
# T_global = None  # To store the clauses


def send_for_unit_propagation(v, mu):
    global last_v, T_global, T
    # Store the last variable assignments and clauses for unit propagation
    last_v = v.clone()
    T_global = T.clone()

def fetch_unit_prop_result():
    global last_v, T_global
    # Perform unit propagation on the last variable assignments
    # and return the new variable assignments
    unit_prop_result = []
    for v_i in last_v:
        v_i_new = v_i.clone()
        changed = True
        while changed:
            changed = False
            # For each clause
            for clause in T_global:
                unassigned_literals = []
                clause_satisfied = False
                for lit in clause:
                    if lit == 0:
                        continue  # Skip padding zeros
                    var_idx = abs(lit) - 1
                    var_value = v_i_new[var_idx].item()
                    if var_value == -1:
                        unassigned_literals.append(lit)
                    elif (lit > 0 and var_value == 1) or (lit < 0 and var_value == 0):
                        clause_satisfied = True
                        break  # Clause is satisfied
                if not clause_satisfied and len(unassigned_literals) == 1:
                    # Unit clause found; assign the unit literal
                    lit = unassigned_literals[0]
                    var_idx = abs(lit) - 1
                    v_i_new[var_idx] = 1 if lit > 0 else 0
                    changed = True  # Changes occurred; need to recheck
        unit_prop_result.append(v_i_new)
    unit_prop_result = torch.stack(unit_prop_result)
    return unit_prop_result
'''

def gather_and_count(v, Q, polarity):
    c = torch.einsum('bv,ckv->bck', v, Q) # (batch_size, num_clause, max_len_clause)
    return c, (((polarity + c) == 2) + ((polarity + c) == -1)).any(dim=-1).sum(dim=-1)

def time_remaining(start_time, time_limit):
    if time.time() - start_time > time_limit:
        return False
    return True

def construct_Q(T, num_var, device):
    Q = torch.zeros(T.shape[0], T.shape[1], num_var, device=device) #(num_clause, len_clause, num_var)
    for clause_idx, clause in enumerate(T):
        for lit_idx, lit in enumerate(clause):
            if lit != 0:
                Q[clause_idx, lit_idx, torch.abs(lit)-1] = 1.0
    return Q

def rbmsat(W_c, b_c, T, B, N, seed, time_limit, upp=100, upw=1, alpha=0.1): # B is batch size, N is the total number of variables
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Initialize variables
    s_max = 0 # initially 0 clauses are satisfied
    v = torch.bernoulli(torch.ones(B, N, device=W_c.device) * 0.5)  # sample random inits 
    # mu = 0.25 * torch.ones_like(v) 
    # Setup tensors
    polarity = torch.sign(T)  # strictly {1, -1}, not 0 (-1 2 -3) -> (-1 1 -1)
    Q = construct_Q(T, formula.num_vars, device) # C * K * N    (num_clause, max_len_clause, num_var)
    W = torch.einsum('ck,kh->ckh', polarity, W_c) # polarity.unsqueeze(1) * W_c.unsqueeze(2) # element-wise multiplication
    # print(W)
    b = b_c.repeat([T.shape[0], 1]) + torch.mm((1 - polarity) / 2, W_c)
    
    # t, d = 1, -1
    step = 0
    s_max_list = []
    best_v = torch.ones(N, device=W_c.device)
    
    start_time = time.time()
    with torch.no_grad():
        while True:
            if time_remaining(start_time, time_limit):
                step += 1
                # if step == 3:
                #     break
                #     # print(f"step {step}, {s_max_list[-1]}")
                
                # if upp > 0 and d == 0:
                #    v_u = torch.vstack([v, fetch_unit_prop_result()])
                #    _, s_u = gather_and_count(v_u, Q, polarity)
                #    ranks = torch.argsort(s_u)
                #    v = v_u[ranks[-B:]]
                #    mu = torch.vstack([mu, mu])[ranks[-B:]]
                # if upp > 0 and t % upp == 0:
                #     d = upw
                #     send_for_unit_propagation(v, mu)
                    
                c, s = gather_and_count(v, Q, polarity)
                s_max_step, idx_max = s.max(dim=0)
                # print(s_max_step.item())
                # print(f"v:{v[idx_max]}")
                if s_max_step.item() > s_max:
                    s_max = s_max_step.item()
                    best_v = v[idx_max].clone()
                    s_max_list.append((step, s_max))
                    print(f"new s_max found {s_max} at step {step}")

                h_logits = b + torch.einsum('bck,ckh->bch', c, W) # 'bck,chk->bhk'  bck,ckh->bch
                
                h = torch.bernoulli(torch.sigmoid(h_logits))
                # print(h[0,0:10])
                ro = torch.sigmoid(torch.einsum('bch,ckh,ckv->bv', h, W, Q)) # 'bhk,chk,ckv->bv'
                # print(f"ro:{ro}")
                # mu = (1 - alpha) * mu + alpha * ro * (1 - ro)
                v = torch.bernoulli(ro)
                
                # t, d = t + 1, max(d - 1, -1)
            else:
                s_max_list.append((step, s_max_list[-1][-1]))
                break
        
    return s_max_list, best_v

T = torch.tensor(formula.clauses, device=device)

bath_size = 128
seed = 42
timeout = 300

alpha = 0.1
upp = 100
upw = 1

s_max, best_v = rbmsat(rbm.W, rbm.b, T, bath_size, formula.num_vars, seed, timeout, upp, upw, alpha)
print(best_v)
x_ax, y_ax = zip(*s_max)
plt.step(x_ax,y_ax)
plt.show()