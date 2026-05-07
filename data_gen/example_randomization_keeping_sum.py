import numpy as np

def vary_with_constant_sum(variables, pct=0.10):
    vars_array = np.array(variables, dtype=float)
    
    # 1. Generate initial random changes within +/- 10%
    deltas = np.random.uniform(-pct, pct, size=len(vars_array)) * vars_array
    
    # 2. & 3. Adjust deltas so they sum to 0
    adjustment = np.sum(deltas) / len(vars_array)
    adjusted_deltas = deltas - adjustment
    
    return vars_array + adjusted_deltas

# Example usage with 10 variables
original = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
new_vars = vary_with_constant_sum(original)