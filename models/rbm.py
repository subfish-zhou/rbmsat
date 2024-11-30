import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils import CNFFormula

num_epochs_config = [{}, 
                     {"num_epochs": 10000, "lr": 0.01, "F_s": -1.0},
                     {"num_epochs": 10000, "lr": 0.01, "F_s": -1.0},
                     {"num_epochs": 10000, "lr": 0.01, "F_s": -1.0},
                     {"num_epochs": 10000, "lr": 0.01, "F_s": -1.0},
                     {"num_epochs": 20000, "lr": 0.01, "F_s": -1.0},
                     {"num_epochs": 50000, "lr": 0.01, "F_s": -1.0},
                     {"num_epochs": 100000, "lr": 0.01, "F_s": -1.0}]

class clauseRBM(nn.Module):
    def __init__(self, clause_length, F_s=None, num_epochs=None, lr=None, device='cpu', verbose=False):
        super(clauseRBM, self).__init__()
        self.n_visible = clause_length
        self.n_hidden = clause_length if clause_length <= 3 else clause_length + 1
        self.device = device
        self.verbose = verbose

        self.F_s = num_epochs_config[clause_length]["F_s"] if F_s is None else F_s
        self.num_epochs = num_epochs_config[clause_length]["num_epochs"] if num_epochs is None else num_epochs
        self.lr = num_epochs_config[clause_length]["lr"] if lr is None else lr
        
        # Initialize parameters
        model_dir='rbm_models'
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f'rbm_length_{clause_length}_F_{self.F_s}_num_epochs_{self.num_epochs}_lr_{self.lr}_device_{device}.pth'
        model_path = os.path.join(model_dir, model_filename)

        if os.path.exists(model_path):
            self.W = nn.Parameter(torch.zeros(self.n_visible, self.n_hidden, device=device))
            self.d = nn.Parameter(torch.zeros(self.n_visible, device=device))
            self.b = nn.Parameter(torch.zeros(self.n_hidden, device=device))
            self.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            # if self.verbose: print(f"Loaded pre-trained RBM from {model_path}")
        else:
            self.train_clause_rbm(F_s=self.F_s, num_epochs=self.num_epochs, lr=self.lr, device=device)
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

            if (epoch+1) % 1000 == 0 and self.verbose:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")


class formulaRBM:
    def __init__(self, 
                 formula: CNFFormula, 
                 F_s=-1.0, 
                 num_epochs=1000, 
                 lr=0.01, 
                 device='cpu', 
                 verbose=False):

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

    def free_energy(self, v):
        return torch.sum(v * self.d_full, dim=1) + torch.sum(torch.log(1 + torch.exp(self.b_full + torch.matmul(v, self.W_full))), dim=1)
    


class formulaRBM_parallel:
    def __init__(self, 
                 formula: CNFFormula, 
                 F_s=-1.0, 
                 num_epochs=1000, 
                 lr=0.01, 
                 device='cpu', 
                 verbose=False):

        literal_clauses = formula.to_literal_form()
        max_clause_length = max(len(clause) for clause in literal_clauses)
        
        # Parallelize over clauses
        C = formula.num_clauses
        K = max_clause_length
        T = torch.zeros(C, K, dtype=torch.int32, device=device)
        H_c = K if K <= 3 else K + 1

        polarities = torch.zeros(C, K, dtype=torch.float32, device=device)  # Stores polarities for each variable in each clause
        for idx, clause in enumerate(literal_clauses):
            for k, (var_idx, is_negated) in enumerate(clause):
                lit = var_idx + 1  # Convert to 1-based index
                if is_negated:
                    lit = -lit
                T[idx, k] = lit
                # Polarity: 1.0 for positive, -1.0 for negated
                polarities[idx, k] = -1.0 if is_negated else 1.0
            # Pad shorter clauses if necessary
            for k in range(len(clause), K):
                T[idx, k] = T[idx, len(clause) - 1]  # Repeat last variable
                polarities[idx, k] = polarities[idx, len(clause) - 1]

        self.W_full = torch.zeros(C, K, H_c, device=device)
        self.b_full = torch.zeros(C, H_c, device=device)
        self.d_full = torch.zeros(C, K, device=device)
        
        for idx in range(C):
            
            # Get pre-trained RBM for this clause length
            rbm = clauseRBM(K, F_s=F_s, num_epochs=num_epochs, lr=lr, device=device, verbose=verbose)
            
            W = rbm.W.detach()  # size n_visible x n_hidden
            b = rbm.b.detach()  # size n_hidden
            d = rbm.d.detach()  # size n_visible
            
            # Adjust W, b, and d according to polarities
            Lambda = torch.diag(polarities[idx]).to(device)  # Shape: [K, K]
            Lambda_diag = torch.diag(Lambda)  # Shape: [K]
            one_minus_Lambda_diag = 1 - Lambda_diag  # Shape: [K]

            # Adjust W: W_prime = Lambda * W
            W_prime = torch.matmul(Lambda, W)  # Shape: [K, H_c]

            # Adjust b: b_prime = b + 0.5 * sum_k [(1 - Lambda_kk) * W_k]
            b_prime = b + 0.5 * ((one_minus_Lambda_diag.unsqueeze(1) * W).sum(dim=0))  # Shape: [H_c]

            # Adjust d: d_prime = Lambda * d
            d_prime = Lambda_diag * d  # Element-wise multiplication, Shape: [K]


            W_prime_padded = torch.zeros(K, H_c, device=device)
            W_prime_padded[:, :H_c] = W_prime  # Copy W_prime into padded tensor

            b_prime_padded = torch.zeros(H_c, device=device)
            b_prime_padded[:H_c] = b_prime  # Copy b_prime into padded tensor

            # Store padded W and b
            self.W_full[idx] = W_prime_padded
            self.b_full[idx] = b_prime_padded
            self.d_full[idx, :C] = d_prime
            # self.W_full[idx] = W_prime
            # self.b_full[idx] = b_prime
            # self.d_full[idx] = d_prime

        self.device = device
        
        self.T = T
        self.polarities = polarities  # Polarity matrix (C x K)
        self.Q = self.construct_Q()  # One-hot encoding (C x N x K)
        self.n_visible = formula.num_vars

        

    def construct_Q(self):
        # T: C x K, values are variable indices with signs
        C, K = self.T.shape
        N = torch.max(torch.abs(self.T)).item()  # Total number of variables
        Q = torch.zeros(C, N, K, device=self.device)
        for c in range(C):
            for k in range(K):
                var_idx = abs(self.T[c, k]) - 1  # Convert to 0-based index
                Q[c, var_idx, k] = 1.0
        return Q

    def sample_h_given_v(self, v):

        c = torch.einsum('cnk,bn->bck', self.Q, v)  # B x C x K
        # Apply polarities
        c = c * self.polarities.unsqueeze(0)  # B x C x K
        # Compute pre-activations
        h_logits = self.b_full.unsqueeze(0) + torch.einsum('bck,ckh->bch', c, self.W_full)
        p_h_given_v = torch.sigmoid(h_logits)
        h_sample = torch.bernoulli(p_h_given_v)
        return h_sample, p_h_given_v

    def sample_v_given_h(self, h):
        # print(f"h shape: {h.shape}")  # Should be [B, C, max_H_c]
        # print(f"self.W_full.transpose(1, 2) shape: {self.W_full.transpose(1, 2).shape}")  # [C, max_H_c, K]

        # h: B x C x H_c
        # Compute contributions from hidden units
        c_logits = torch.einsum('bch,ckh->bck', h, self.W_full.transpose(1, 2))  # B x C x K
        # Adjust for polarities
        c_logits = c_logits * self.polarities.unsqueeze(0)
        # Add visible biases
        c_logits = c_logits + self.d_full.unsqueeze(0)  # B x C x K
        # Scatter-add to variables
        v_logits = torch.einsum('bck,cnk->bn', c_logits, self.Q)  # B x N x K
        p_v_given_h = torch.sigmoid(v_logits)
        v_sample = torch.bernoulli(p_v_given_h)
        return v_sample, p_v_given_h
    
    def gibbs_sampling(self, v_init, k=1):
        v = v_init
        for _ in range(k):
            h_sample, _ = self.sample_h_given_v(v)
            v_sample, _ = self.sample_v_given_h(h_sample)
            v = v_sample
        return v