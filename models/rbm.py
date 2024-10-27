import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils.cnf_parser import CNFFormula

class clauseRBM(nn.Module):
    def __init__(self, clause_length, F_s=-1.0, num_epochs=1000, lr=0.01, device='cpu', verbose=False):
        super(clauseRBM, self).__init__()
        self.n_visible = clause_length
        self.n_hidden = clause_length if clause_length <= 3 else clause_length + 1
        self.device = device
        self.verbose = verbose
        
        # Initialize parameters
        model_dir='rbm_models'
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f'rbm_length_{clause_length}_F_{F_s}.pth'
        model_path = os.path.join(model_dir, model_filename)

        if os.path.exists(model_path):
            self.W = nn.Parameter(torch.zeros(self.n_visible, self.n_hidden, device=device))
            self.d = nn.Parameter(torch.zeros(self.n_visible, device=device))
            self.b = nn.Parameter(torch.zeros(self.n_hidden, device=device))
            self.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            if self.verbose: print(f"Loaded pre-trained RBM from {model_path}")
        else:
            self.train_clause_rbm(F_s=F_s, num_epochs=num_epochs, lr=lr, device=device)
            torch.save(self.state_dict(), model_path)
            if self.verbose: print(f"Saved pre-trained RBM to {model_path}")
        
    def free_energy(self, v):
        # v: batch_size x n_visible
        v_term = torch.matmul(v, self.d)
        pre_activation = self.b + torch.matmul(v, self.W)
        h_term = torch.sum(torch.log1p(torch.exp(pre_activation)), dim=1)
        F_v = -v_term - h_term
        return F_v  # batch_size
        
    def train_clause_rbm(self, F_s=-1.0, num_epochs=1000, lr=0.01, device='cpu'):
        """
        Trains an RBM for clauses of a given length.
        Args:
            clause_length: Length of the clause
            F_s: Target free energy for non-zero assignments
            num_epochs: Number of training epochs
            lr: Learning rate
            device: Device to run the computation on ('cpu' or 'cuda')
        Returns:
            Trained RBM
        """
        self.W = nn.Parameter(torch.randn(self.n_visible, self.n_hidden, device=device) * 0.1)
        self.d = nn.Parameter(torch.zeros(self.n_visible, device=device))
        self.b = nn.Parameter(torch.zeros(self.n_hidden, device=device))

        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Generate all possible binary vectors
        n_samples = 2 ** self.n_visible
        v_vectors = torch.tensor([ [int(x) for x in bin(i)[2:].zfill(self.n_visible)] for i in range(n_samples) ], dtype=torch.float32, device=device)
        # Target free energies
        F_targets = torch.zeros(n_samples, device=device)
        for i in range(n_samples):
            if v_vectors[i].sum() == 0:
                F_targets[i] = 0.0
            else:
                F_targets[i] = F_s
        
        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            F_v = self.free_energy(v_vectors)
            loss = torch.mean((F_v - F_targets) ** 2)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 200 == 0 and self.verbose:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")


class formulaRBM:
    def __init__(self, formula: CNFFormula, F_s=-1.0, num_epochs=1000, lr=0.01, device='cpu', verbose=False):

        literal_clauses = formula.to_literal_form()
        n_visible_total = formula.num_vars
        n_hidden_total = 0
        for clause in literal_clauses:
            clause_length = len(clause)
            n_hidden = clause_length if clause_length <= 3 else clause_length + 1
            n_hidden_total += n_hidden

        self.W_full = torch.zeros(n_visible_total, n_hidden_total, device=device)
        self.b_full = torch.zeros(n_hidden_total, device=device)
        self.d_full = torch.zeros(n_visible_total, device=device)
        
        curr_hidden_idx = 0

        for clause in literal_clauses:
            # clause: list of (var_idx, is_negated)
            variable_indices, signs = zip(*clause)
            # variable_indices = [var_idx for (var_idx, is_negated) in clause]
            # signs = [is_negated for (var_idx, is_negated) in clause]
            
            clause_length = len(clause)
            n_visible = clause_length
            n_hidden = n_visible if n_visible <= 3 else n_visible + 1
            
            # Get pre-trained RBM for this clause length
            rbm = clauseRBM(clause_length, F_s=F_s, num_epochs=num_epochs, lr=lr, device=device, verbose=verbose)
            
            # Adjust W and b according to signs
            Lambda = torch.tensor([-1.0 if is_negated else 1.0 for is_negated in signs], device=device)
            W = rbm.W.detach()  # size n_visible x n_hidden
            b = rbm.b.detach()  # size n_hidden
            d = rbm.d.detach()  # size n_visible
            
            W_prime = Lambda.unsqueeze(1) * W  # size n_visible x n_hidden
            b_prime = b + 0.5 * torch.matmul((1 - Lambda), W)
            
            # Map W_prime into W_full
            for local_var_idx, var_idx in enumerate(variable_indices):
                self.W_full[var_idx, curr_hidden_idx : curr_hidden_idx + n_hidden] += W_prime[local_var_idx, :]
                self.d_full[var_idx] += d[local_var_idx]
            # Map b_prime into b_full
            self.b_full[curr_hidden_idx : curr_hidden_idx + n_hidden] = b_prime
            
            curr_hidden_idx += n_hidden

        self.n_visible = n_visible_total
        self.n_hidden = n_hidden_total
        self.device = device
        
    def sample_h_given_v(self, v):
        # v: batch_size x n_visible
        p_h_given_v = torch.sigmoid(self.b_full + torch.matmul(v, self.W_full))
        h_sample = torch.bernoulli(p_h_given_v)
        return h_sample, p_h_given_v
    
    def sample_v_given_h(self, h):
        # h: batch_size x n_hidden
        p_v_given_h = torch.sigmoid(self.d_full + torch.matmul(h, self.W_full.t()))
        v_sample = torch.bernoulli(p_v_given_h)
        return v_sample, p_v_given_h
    
    def gibbs_sampling(self, v_init, k=1):
        v = v_init
        for _ in range(k):
            h_sample, _ = self.sample_h_given_v(v)
            v_sample, _ = self.sample_v_given_h(h_sample)
            v = v_sample
        return v