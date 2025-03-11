#%% 
import sympy as sp
import numpy as np

# Define symbolic variables
lambda_ = sp.symbols('lambda')
Delta = sp.symbols('delta')
L = sp.symbols('L')

n=5

# Generate a L matrix n by n
L = np.random.rand(n, n)
for i in range(n):
    # Sum all elements in the column except for the diagonal element (matrix[i, i])
    column_sum = np.sum(L[:, i]) - L[i, i]
    L[i, i] = -column_sum  # Update the diagonal element
print('L matrix:', L)
# Create a Delta matrix of zeros with shape (n, n)
Delta = np.zeros((n, n))

scaling_factor = 0.1
# Assign a random value to the last entry (bottom-right corner)
random_value = np.random.rand()  # You can modify this to any range or value
Delta[n-1, n-1] = random_value*scaling_factor

print('Delta Matrix:', Delta)

# Generate the identity matrix
identity_matrix = sp.eye(n)

# Define the matrix ((L+Delta) - lambda * I)
matrix_lambda_I = (L + Delta) - lambda_ * identity_matrix

#Obtaining eigenvalue without removing higher order terms
eigenvalues = np.linalg.eigvals(L + Delta)

print('eigval for lamda^(n):', eigenvalues)
# Get the characteristic polynomial by calculating the determinant

#Obtaining eigenvalue with removing higher order terms
polynomial = matrix_lambda_I.det()

simplified_polynomial = sp.simplify(polynomial)

# Check if the simplified expression is a rational function (i.e., fraction)
numerator, denominator = sp.fraction(simplified_polynomial)

# Now, treat the numerator and denominator separately as polynomials
# Simplify both numerator and denominator
simplified_numerator = sp.simplify(numerator)
simplified_denominator = sp.simplify(denominator)

# You can treat the denominator as a polynomial if needed
denom_poly = sp.Poly(simplified_denominator, lambda_)

# Create a polynomial object for the numerator if needed (useful for eigenvalue calculation)
numerator_poly = sp.Poly(simplified_numerator, lambda_)

# Set the maximum degree of lambda to remove higher order terms
max_degree = 1

# Extract terms of the numerator polynomial
numerator_terms = numerator_poly.terms()

# Remove higher-order terms beyond the specified maximum degree
truncated_numerator = sum(coef * lambda_**deg[0] for deg, coef in numerator_terms if deg[0] <= max_degree)

# Solve for the largest eigenvalue (after truncating higher-order terms)
largest_eigenvalue = sp.solve(truncated_numerator, lambda_)

# Print the result
print('eigval for lamda^(1):', largest_eigenvalue)

# Function to compute the cofactor of an element in a matrix
def cofactor(matrix, i, j):
    submatrix = np.delete(matrix, i, axis=0)  # Remove the i-th row
    submatrix = np.delete(submatrix, j, axis=1)  # Remove the j-th column
    return (-1) ** (i + j) * np.linalg.det(submatrix)

# Function to compute the desired summation
def summation(Delta, L):
    n = Delta.shape[0]  # Assume square matrices
    sum_top = 0
    sum_top_1 = 0  # Initialize sum_top_1

    # List to store cofactor sums
    cofactor_sum = [] 
    cofactor_of_sum_1 = []

    # Calculate matrix sum
    matrix_sum = L + Delta

    for i in range(n):
        # Diagonal elements (Delta__ii from Delta_ matrix)
        Delta__ii = Delta[i, i]
        L_ii = L[i, i]

        # Calculating sum of cofactors for the function f and g
        cofactor_of_sum = cofactor(matrix_sum, i, i)  
        cofactor_sum.append(cofactor_of_sum)
        
        cofactor_of_sum_1_value = cofactor(L, i, i) 
        cofactor_of_sum_1.append(cofactor_of_sum_1_value)

        # Obtain function f 
        sum_top += Delta__ii * cofactor_of_sum   
        # Obtain function g
        sum_top_1 += Delta__ii * cofactor_of_sum_1_value   

    # If the bottom sum is zero, return an error or handle it as needed
    if sum(cofactor_sum) != 0 and sum(cofactor_of_sum_1) != 0:
        result1 = sum_top / sum(cofactor_sum)
        result2 = sum_top_1 / sum(cofactor_of_sum_1)
        print('fun f:', result1)
        print('fun g:', result2)
    else:
        return float('inf')  # Or handle this case appropriately

# Calculate the summation
summation(Delta, L)

# %%
