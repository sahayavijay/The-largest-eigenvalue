#%%
import sympy as sp 
import numpy as np
lambda_ = sp.symbols('lambda')
Delta_=sp.symbols('delta')
L=sp.symbols('L')
"""
Try writing a code in this format to check 
"""

def decompose_M(M):
    return L, Delta

def analytical_formula_type_1(L, Delta):
    lambda0 = ... some calculations
    return lambda0

def comparison(M)
    L,Delta = decompose_M(M)
    true_lambda0 = eigval_M(M)
    analytical_lambda0 = analytical_formula_type_1(L,Delta)
    return true_lambda0, analytical_lambda0
N = 2
M = make some random NxN matrix
comparison(M)
#%%


#%%
import sympy as sp 
import numpy as np
from fractions import Fraction
lambda_ = sp.symbols('lambda')
Delta_=sp.symbols('delta')
L=sp.symbols('L')
def eigval_M(n):
    random_matrix = np.random.rand(n, n)
    random_matrix_sympy = sp.Matrix(random_matrix)
    identity_matrix = sp.eye(n)
    # Define the matrix (A - lambda * I)
    matrix_lambda_I = random_matrix_sympy - lambda_ * identity_matrix
    # Calculate the determinant of (A - lambda * I)
    polynomial = matrix_lambda_I.det()
    # Simplify the polynomial to avoid complex rational expressions
    simplified_polynomial = sp.simplify(polynomial)
    # Expand the polynomial (this may help to avoid division by lambda)
    expanded_polynomial = sp.expand(simplified_polynomial)
    max_degree=1
    poly_terms = sp.Poly(expanded_polynomial, lambda_).terms()
    truncated_polynomial = sum(coef * lambda_**deg[0] for deg, coef in poly_terms if deg[0] <= max_degree)
    largest_eigenvalue = sp.solve(truncated_polynomial, lambda_)
    print("Random matrix (A):", random_matrix)
    print("Matrix (A - lambda * I):", matrix_lambda_I)
    print("Determinant of (A - lambda * I):", expanded_polynomial)
    print("Truncated Polynomial:", truncated_polynomial)
    print(f"The largest eigenvalue is: {largest_eigenvalue[0]}")
    
eigval_M(3)


# Function to compute the cofactor of an element in a matrix
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
        sum_L_and_Delta_ = matrix_L[i, i] + matrix_Delta_[i, i]

        # Cofactors
        cofactor_L_ii = cofactor(matrix_L, i, i)
        cofactor_of_sum = cofactor(matrix_sum, i, i)

        # Update the summation terms
        sum_top += Delta__ii * cofactor_L_ii
        sum_bottom += cofactor_of_sum
        
    # If the bottom sum is zero, we return an error or handle it as needed
    if sum_bottom != 0:
        return sum_top / sum_bottom
    else:
        return float('inf')  # Or handle this case appropriately

# Function to convert decimal to fraction
def decimal_to_fraction(decimal_number):
    # Convert the decimal number to a Fraction
    return Fraction(decimal_number).limit_denominator()  # limit_denominator helps to simplify the fraction
# Example matrices (replace with your own)

matrix_Delta_ = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 1]])
matrix_L = np.array([[-1,0, 2],[1,-1, 0],[0,1, -2]])
matrix_sum = matrix_L + matrix_Delta_
# Calculate the summation
result = summation(matrix_Delta_, matrix_L)
fraction = decimal_to_fraction(result)
print("Result:", fraction)
#%%

