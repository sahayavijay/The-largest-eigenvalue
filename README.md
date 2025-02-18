#%%
import sympy as sp 
import numpy as np
lambda_ = sp.symbols('lambda')
Delta_=sp.symbols('delta')
L=sp.symbols('L')

def eigval_M():
    M = sp.Matrix([[-1-lambda_, 2], [1, -1-lambda_]])
    polynomial=M.det()
    max_degree=1
    poly_terms = sp.Poly(polynomial, lambda_).terms()
    truncated_polynomial = sum(coef * lambda_**deg[0] for deg, coef in poly_terms if deg[0] <= max_degree)
    largest_eigenvalue = sp.solve(truncated_polynomial, lambda_)
    print(M)
    print("Polynomial:",polynomial)
    print("Truncated Polynomial:",truncated_polynomial)
    print(f"The largest eigenvalue is: {largest_eigenvalue[0]}")
    
eigval_M()


### Function to compute the cofactor of an element in a matrix
def cofactor(matrix, i, j):
    submatrix = np.delete(matrix, i, axis=0)  # Remove the i-th row
    submatrix = np.delete(submatrix, j, axis=1)  # Remove the j-th column
    return (-1) ** (i + j) * np.linalg.det(submatrix)

# Function to compute the desired summation
def summation(matrix_Delta_, matrix_L):
    n = matrix_Delta_.shape[0]  # Assume square matrices
    sum_top = 0
    sum_bottom = 0
    
    for i in range(n):
        # Diagonal elements (Delta__ii and L_ii)
        Delta__ii = matrix_Delta_[i, i]
        L_ii = matrix_L[i, i]
        
        # Cofactors
        cofactor_L_ii = cofactor(matrix_L, i, i)
        
        # Update the summation terms
        sum_top += Delta__ii * cofactor_L_ii
        sum_bottom += cofactor_L_ii
    # If the bottom sum is zero, we return an error or handle it as needed
    if sum_bottom != 0:
        return sum_top / sum_bottom
    else:
        return float('inf')  # Or handle this case appropriately

# Example matrices (replace with your own)
matrix_Delta_ = np.array([[0, 0],[0, 1]])
matrix_L = np.array([[-1,2],[1,-2],])

# Calculate the summation
result = summation(matrix_Delta_, matrix_L)
print("Result:", result)


# %%

