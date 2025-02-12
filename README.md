# The-largest-eigenvalue
import sympy as sp 
lambda_ = sp.symbols('lambda')

def matrix_M():
    M = sp.Matrix([[-1-lambda_, 0, 0, 1], [1, -1-lambda_, 0, 0], [1, 1, -1-lambda_, 0], [0, 0, 1, -1-lambda_]])
    polynomial=M.det()
    max_degree=1
    poly_terms = sp.Poly(polynomial, lambda_).terms()
    truncated_polynomial = sum(coef * lambda_**deg[0] for deg, coef in poly_terms if deg[0] <= max_degree)
    largest_eigenvalue = sp.solve(truncated_polynomial, lambda_)
    print("Matrix:",M)
    print("Polynomial:",polynomial)
    print("Truncated Polynomial:",truncated_polynomial)
    print(f"The largest eigenvalue is: {largest_eigenvalue[0]}")
    
matrix_M()
