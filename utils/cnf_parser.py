from typing import List, Tuple

class CNFFormula:
    def __init__(self):
        self.num_vars = 0
        self.num_clauses = 0
        self.clauses: List[List[int]] = []  # [[1, -2], [3, -4, 5], ...]
        
    @staticmethod
    def from_dimacs(dimacs_str: str) -> 'CNFFormula':

        formula = CNFFormula()
        lines = dimacs_str.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('c'):  # comments
                continue
            if line.startswith('p'):  # problem declaration
                _, _, num_vars, num_clauses, _ = line.split()
                formula.num_vars = int(num_vars)
                formula.num_clauses = int(num_clauses)
            elif line:  # clause
                clause = [int(x) for x in line.split()[1:-1]]  # discard weight and 0
                formula.clauses.append(clause)
                
        return formula
    
    def to_literal_form(self) -> List[List[Tuple[int, bool]]]:
        """
        (var_idx, is_negated)
        """
        literal_clauses = []
        for clause in self.clauses:
            literal_clause = []
            for lit in clause:
                var_idx = abs(lit) - 1 # 0-based
                is_negated = lit < 0
                literal_clause.append((var_idx, is_negated))
            literal_clauses.append(literal_clause)
        return literal_clauses
