import numpy as np

def gram_schmidt(vectors):
    orthonormal_vectors = []
    for v in vectors:
        ortho_v = v.copy()
        for prev_v in orthonormal_vectors:
            proj = np.dot(v, prev_v) * prev_v
            ortho_v -= proj
        norm = np.linalg.norm(ortho_v)
        if norm > 1e-10:
            ortho_v /= norm
        orthonormal_vectors.append(ortho_v)
    return np.array(orthonormal_vectors)

n = np.arange(0, 11)
original_vectors = np.array([
    np.ones_like(n),  
    n,               
    n**2,             
    n**3,
    n**4             
], dtype=float)

# Apply Gram-Schmidt to the basis
orthonormal_basis = gram_schmidt(original_vectors)

# Output the orthonormal vectors
# print(original_vectors)
print(orthonormal_basis)
