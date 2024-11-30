import numpy as np
import numpy.polynomial.polynomial as poly

def inner_product(f1, f2, a, b):
    """Compute inner product of two functions over interval [a,b]"""
    def integrand(x):
        return f1(x) * f2(x)
    return np.quad(integrand, a, b)[0]

def gram_schmidt(basis, a, b):
    """Perform Gram-Schmidt orthonormalization"""
    orthogonal_basis = []
    orthonormal_basis = []
    
    for f in basis:
        # Project out previous orthogonal vectors
        proj = np.zeros_like(f)
        for u in orthogonal_basis:
            ip = inner_product(f, u, a, b)
            proj += ip * u
        
        # Compute orthogonal vector
        v = f - proj
        orthogonal_basis.append(v)
        
        # Normalize
        norm = np.sqrt(inner_product(v, v, a, b))
        orthonormal_basis.append(lambda x, v=v, norm=norm: v(x)/norm)
    
    return orthonormal_basis

# Define original basis functions
def f0(x): return 1
def f1(x): return x
def f2(x): return x**2
def f3(x): return x**3
def f4(x): return x**4

basis = [f0, f1, f2, f3, f4]
a, b = 0, 10

orthonormal_set = gram_schmidt(basis, a, b)

# Verify orthonormality
print("Orthonormality Check:")
for i in range(len(orthonormal_set)):
    for j in range(len(orthonormal_set)):
        ip = inner_product(orthonormal_set[i], orthonormal_set[j], a, b)
        print(f"IP({i},{j}) = {ip:.6f}")