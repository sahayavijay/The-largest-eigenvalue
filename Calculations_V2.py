#%%
import numpy as np
import sympy as sp
from fractions import Fraction
lambda_ = sp.symbols('lambda')
Delta_=sp.symbols('delta')
L=sp.symbols('L')

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
      
# Generate random matrices
x1=np.random.rand(2,2)  # For reproducibility, you can remove this for true randomness
size = 2  # Define the size of the matrix (e.g., 3x3)
random_matrix_sympy = x1
identity_matrix = sp.eye(2)

# Define the matrix (A - lambda * I)
matrix_lambda_I = random_matrix_sympy-lambda_*identity_matrix

# Calculate the determinant of (A - lambda * I)
polynomial = matrix_lambda_I.det()

max_degree=1
poly_terms = sp.Poly(polynomial, lambda_).terms()
truncated_polynomial = sum(coef * lambda_**deg[0] for deg, coef in poly_terms if deg[0] <= max_degree)
largest_eigenvalue = sp.solve(truncated_polynomial, lambda_)

matrix_Delta_ = np.array([[0,0],[0,0.1]])  # Random matrix for Delta_
random_ = x1 # Random matrix for L, can include negative values
matrix_L=random_ 
row, col = 1,1
matrix_L[row, col] -= 0.1

# Summing both matrices
matrix_sum = matrix_L + matrix_Delta_ 

# Calculate the summation
result = summation(matrix_Delta_, matrix_L)

# Function to convert decimal to fraction
def decimal_to_fraction(decimal_number):
    # Convert the decimal number to a Fraction
    return Fraction(decimal_number).limit_denominator()  # limit_denominator helps to simplify the fraction

# Convert the result to a fraction
fraction = decimal_to_fraction(result)

# Output the result
print("largest eigen value by numerical calculation:", largest_eigenvalue) #Largest eigen value lambda_0 by numerical calculation
print("Matrix_Delta:", matrix_Delta_) #Random Matrix L
print("Matrix_L:", matrix_L) #Matrix Delta
print("largest eigen value using formula:",result) #largest eigen value lambda_0 using formula
# %%
