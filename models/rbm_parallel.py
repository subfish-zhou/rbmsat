import torch
from utils.cnf_parser import CNFFormula
from .rbm import clauseRBM

class formulaRBM_parallel:
    def __init__(self, formula: CNFFormula, F_s=-1.0, num_epochs=1000, lr=0.01, device='cpu', verbose=False):

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

            self.W_full[idx] = W_prime
            self.b_full[idx] = b_prime
            self.d_full[idx] = d_prime

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
        # v: B x N
        # Gather variables for clauses
        c = torch.einsum('cnk,bn->bck', self.Q, v)  # B x C x K
        # Apply polarities
        c = c * self.polarities.unsqueeze(0)  # B x C x K
        # Compute pre-activations
        h_logits = self.b_full.unsqueeze(0) + torch.einsum('bck,ckh->bch', c, self.W_full)
        p_h_given_v = torch.sigmoid(h_logits)
        h_sample = torch.bernoulli(p_h_given_v)
        return h_sample, p_h_given_v

    def sample_v_given_h(self, h):
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