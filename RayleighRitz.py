from numpy import matmul
from scipy.linalg import eigh, cholesky, solve_triangular
from numpy.linalg import cond, qr
from numpy import empty, diag, eye, ones, zeros, finfo

def RR1(X, AX, BX):
  """
  Computes m least dominant generalized eigenpairs of 
  (A,B) with respect to the range of X using a 
  Rayleigh-Ritz projection.

  Parameters:
  - X, AX, BX : Input matrices

  Returns:
  - hX     : Eigenvectors
  - Lambda : Eigenvalues
  """
  
  G1 = matmul(X.T, AX)
  G2 = matmul(X.T, BX)
    
  # Find L s.t. L @ L.T = G2
  L = cholesky(G2, lower=True)

  # G2[:] = L^(-1) @ G1 @ L^(-T)
  G2[:] = solve_triangular(L, G1, lower=True, overwrite_b=True)
  G2[:] = solve_triangular(L, G2.T, lower=True)

  Lambda, hY = eigh(G2)

  # hX = L^(-T) @ hY
  hX = solve_triangular(L, hY, trans='T', lower=True)

  return hX, Lambda

def RR2(X, Z, AX, AZ, BZ, check_L_cond=False, Lambda=None):
  """
  Computes m least dominant generalized eigenpairs of 
  (A,B) with respect to the range of [X,Z] using a 
  Rayleigh-Ritz projection.

  Parameters:
  - X             : Input matrix such that X.T @ BX = I
  - Z, AX, AZ, BZ : Additional input matrices
  - check_L_cond  : Boolean to check condition number
  - Lambda        : Optional vector of previous eigenvalues

  Returns:
  - hX     : Eigenvectors
  - Lambda : Eigenvalues
  - Condition check result (if check_L_cond=True)
  - VtBV   : Optional return if condition fails
  """

  tau = 2 * finfo(float).eps

  _, m = X.shape
  _, q = Z.shape
  G1 = empty((m + q, m + q))
  G2 = empty((m + q, m + q))
  VtBV = empty((m + q, m + q))
  hX = empty((m + q, m))

  # G1[:] = [X, Z].T @ [AX, AZ]
  if Lambda is None or len(Lambda) == 0:
    G1[:m, :m] = matmul(X.T, AX)
  else:
    G1[:m, :m] = diag(Lambda)
  G1[:m, m:m+q] = matmul(X.T, AZ)
  G1[m:m+q, m:m+q] = matmul(Z.T, AZ)
  G1[m:m+q, :m] = G1[:m, m:m+q].T

  # VtBV[:] = [X, Z].T @ [BX, BZ]
  VtBV[:m, :m] = eye(m)
  VtBV[:m, m:m+q] = matmul(X.T, BZ)
  VtBV[m:m+q, m:m+q] = matmul(Z.T, BZ)
  VtBV[m:m+q, :m] = VtBV[:m, m:m+q].T

  D = diag(diag(VtBV) ** (-0.5))
  
  # Find L such that L @ L.T = D @ VtBV @ D
  matmul(D, matmul(VtBV, D), out=G2) # G2[:] = D @ VtBV @ D
  L = cholesky(G2, lower=True)

  if check_L_cond:
    if cond(L) ** (-3) < tau:
      return hX, zeros(m), False, VtBV

  # G2[:] = L^(-1) @ D @ G1 @ D @ L^(-T)
  matmul(D, matmul(G1, D), out=G2) # G2[:] = D @ G1 @ D
  G2[:] = solve_triangular(L, G2, lower=True, overwrite_b=True)
  G2[:] = solve_triangular(L, G2.T, lower=True)

  Lambda, hY = eigh(G2)

  # hX = D @ L^(-T) @ hY
  hX[:] = solve_triangular(L, hY[:, :m], trans='T', lower=True)
  hX[:] = matmul(D, hX)

  if check_L_cond:
    return hX, Lambda[:m], True, VtBV
  else:
    return hX, Lambda[:m]

def RR3(X, Z, P, AX, AZ, AP, BZ, BP, check_L_cond=False, Lambda=None):
  """
  Computes m least dominant generalized eigenpairs of 
  (A,B) with respect to the range of [X,Z,P] using a 
  Rayleigh-Ritz projection.

  Parameters:
  - X                        : Input matrix such that X.T @ BX = I
  - Z, P, AX, AZ, AP, BZ, BP : Additional input matrices
  - check_L_cond             : Boolean to check condition number
  - Lambda                   : Optional vector of previous eigenvalues

  Returns:
  - hX     : Eigenvectors
  - Lambda : Eigenvalues
  - Condition check result (if check_L_cond=True)
  - VtBV   : Optional return if condition fails
  """
    
  tau = 2 * finfo(float).eps
  _, m = X.shape
  _, q = Z.shape
  
  G1 = empty((m + 2*q, m + 2*q))
  G2 = empty((m + 2*q, m + 2*q))
  VtBV = empty((m + 2*q, m + 2*q))
  hX = empty((m + 2*q, m))
  
  # G1[:] = [X, Z, P].T @ [AX, AZ, AP]
  if Lambda is None or len(Lambda) == 0:
    G1[:m, :m] = matmul(X.T, AX)
  else:
    G1[:m, :m] = diag(Lambda)
  G1[:m, m:m+q] = matmul(X.T, AZ)
  G1[:m, m+q:m+2*q] = matmul(X.T, AP)
  G1[m:m+q, m:m+q] = matmul(Z.T, AZ)
  G1[m:m+q, m+q:m+2*q] = matmul(Z.T, AP)
  G1[m+q:m+2*q, m+q:m+2*q] = matmul(P.T, AP)
  G1[m:m+q, :m] = G1[:m, m:m+q].T
  G1[m+q:m+2*q, :m] = G1[:m, m+q:m+2*q].T
  G1[m+q:m+2*q, m:m+q] = G1[m:m+q, m+q:m+2*q].T
  
  # VtBV[:] = [X, Z, P].T @ [BX, BZ, BP]
  VtBV[:m, :m] = eye(m)
  VtBV[:m, m:m+q] = matmul(X.T, BZ)
  VtBV[:m, m+q:m+2*q] = matmul(X.T, BP)
  VtBV[m:m+q, m:m+q] = matmul(Z.T, BZ)
  VtBV[m:m+q, m+q:m+2*q] = matmul(Z.T, BP)
  VtBV[m+q:m+2*q, m+q:m+2*q] = matmul(P.T, BP)
  VtBV[m:m+q, :m] = VtBV[:m, m:m+q].T
  VtBV[m+q:m+2*q, :m] = VtBV[:m, m+q:m+2*q].T
  VtBV[m+q:m+2*q, m:m+q] = VtBV[m:m+q, m+q:m+2*q].T

  D = diag(diag(VtBV) ** (-0.5))

  # Find L such that L @ L.T = D @ VtBV @ D
  matmul(D, matmul(VtBV, D), out=G2) # G2[:] = D @ VtBV @ D
  L = cholesky(G2, lower=True) 

  if check_L_cond:
    if cond(L) ** (-3) < tau:
      return hX, zeros(m), False, VtBV

  # G2[:] = L^(-1) @ D @ G1 @ D @ L^(-T)
  matmul(D, matmul(G1, D), out=G2) # G2[:] = D @ G1 @ D
  G2[:] = solve_triangular(L, G2, lower=True, overwrite_b=True)
  G2[:] = solve_triangular(L, G2.T, lower=True)

  Lambda, hY = eigh(G2)
  
  # hX = D @ L^(-T) @ hY
  hX[:] = solve_triangular(L, hY[:, :m], trans='T', lower=True)
  hX[:] = matmul(D, hX)

  if check_L_cond:
    return hX, Lambda[:m], True, VtBV
  else:
    return hX, Lambda[:m]

def RR4(X, Z, AX, AZ, Lambda=None, modified=False):
  """
  Computes m least dominant generalized eigenpairs of 
  (A,B) with respect to the range of [X, Z] using a 
  Rayleigh-Ritz projection.

  Parameters:
  - X, Z   : Input matrices such that [X, Z].T @ B @ [X, Z] = I
  - AX, AZ : Additional input matrices
  - Lambda : Optional vector of previous eigenvalues

  Returns:
  - hZ[:, :m]  : Eigenvectors
  - Lambda[:m] : Eigenvalues
  """
    
  _, m = X.shape
  _, q = Z.shape
  
  G = empty((m + q, m + q))
  
  # G1[:] = [X, Z].T @ [AX, AZ]
  if Lambda is None or len(Lambda) == 0:
    G[:m, :m] = matmul(X.T, AX)
  else:
    G[:m, :m] = diag(Lambda)    
  G[:m, m:m+q] = matmul(X.T, AZ)
  G[m:m+q, m:m+q] = matmul(Z.T, AZ)
  G[m:m+q, :m] = G[:m, m:m+q].T
    
  Lambda, hZ = eigh(G)

  if modified:
    hQt, _ = qr(hZ[:m, m:].T)
    hY = matmul(hZ[:, m:], hQt)
    return hZ[:, :m], Lambda[:m], hY

  return hZ[:, :m], Lambda[:m]

def RR5(X, Z, P, AX, AZ, AP, Lambda=None, modified=False):
  """
  Computes m least dominant generalized eigenpairs of 
  (A,B) with respect to the range of [X, Z, P] using a 
  Rayleigh-Ritz projection.

  Parameters:
  - X, Z, P    : Input matrices such that [X, Z, P].T @ B @ [X, Z, P] = I
  - AX, AZ, AP : Additional input matrices
  - Lambda     : Optional vector of previous eigenvalues
    
  Returns:
  - hZ[:, :m]  : Eigenvectors
  - Lambda[:m] : Eigenvalues
  """ 

  _, m = X.shape
  _, q = Z.shape
    
  G = empty((m + 2*q, m + 2*q))

  # G[:] = [X, Z, P].T @ [AX, AZ, AP]
  if Lambda is None or len(Lambda) == 0:
    G[:m, :m] = matmul(X.T, AX)
  else:
    G[:m, :m] = diag(Lambda)
  G[:m, m:m+q] = matmul(X.T, AZ)
  G[:m, m+q:m+2*q] = matmul(X.T, AP)
  G[m:m+q, m:m+q] = matmul(Z.T, AZ)
  G[m:m+q, m+q:m+2*q] = matmul(Z.T, AP)
  G[m+q:m+2*q, m+q:m+2*q] = matmul(P.T, AP)
  G[m:m+q, :m] = G[:m, m:m+q].T
  G[m+q:m+2*q, :m] = G[:m, m+q:m+2*q].T
  G[m+q:m+2*q, m:m+q] = G[m:m+q, m+q:m+2*q].T

  Lambda, hZ = eigh(G)

  if modified:
    hQt, _ = qr(hZ[:m, m:].T)
    hY = matmul(hZ[:, m:], hQt)
    return hZ[:, :m], Lambda[:m], hY

  return hZ[:, :m], Lambda[:m]

def RR6(X, AX):
  """
  Computes k least dominant generalized eigenpairs of 
  (A, B) with respect to the range of X using a 
  Rayleigh-Ritz projection.

  Parameters:
  - X, AX : Input matrices
  
  Returns:
  - hX     : Eigenvectors
  - Lambda : Eigenvalues
  """
    
  G = matmul(X.T, AX)
    
  Lambda, hX = eigh(G) 
    
  return hX, Lambda

def RR_BLOPEX1(X, Z, AX, AZ, BZ, Lambda=None):
  """
  Computes m least dominant generalized eigenpairs of 
  (A, B) with respect to the range of [X, Z] using a 
  Rayleigh-Ritz projection.

  Parameters:
  - X          : Input matrix such that X.T @ BX = I
  - Z          : Input matrix such that Z.T @ BZ = I
  - AX, AZ, BZ : Additional input matrices
  - Lambda     : Optional vector of previous eigenvalues

  Returns:
  - hX         : Eigenvectors
  - Lambda[:m] : Eigenvalues
  """
    
  _, m = X.shape
    
  G1 = empty((2*m, 2*m))
  G2 = empty((2*m, 2*m))
  hX = empty((2*m, m))
    
  # G1[:] = [X, Z].T @ [AX, AZ]
  if Lambda is None or len(Lambda) == 0:
    G1[:m, m:] = matmul(X.T, AX)
  else:
    G1[:m, :m] = diag(Lambda)
  G1[:m, m:] = matmul(X.T, AZ)
  G1[m:, m:] = matmul(Z.T, AZ)
  G1[m:, :m] = G1[:m, m:].T  

  # G2[:] = [X, Z].T @ [BX, BZ]
  G2[:m, :m] = diag(ones(m))
  G2[:m, m:] = matmul(X.T, BZ)
  G2[m:, m:] = diag(ones(m))
  G2[m:, :m] = G2[:m, m:].T 

  # Find L such that L @ L.T = G2
  L = cholesky(G2, lower=True)
  
  # G2[:] = L^(-1) @ G1 @ L^(-T)
  G2[:] = solve_triangular(L, G1, lower=True, overwrite_b=True)
  G2[:] = solve_triangular(L, G2.T, lower=True)
    
  Lambda, hY = eigh(G2)
    
  # hX = L^(-T) @ hY[:, :m]
  hX = solve_triangular(L, hY[:, :m], trans='T', lower=True)
  
  return hX, Lambda[:m]

def RR_BLOPEX2(X, Z, P, AX, AZ, AP, BZ, BP, Lambda=None):
  """
  Computes m least dominant generalized eigenpairs of 
  (A, B) with respect to the range of [X, Z, P] using a 
  Rayleigh-Ritz projection.

  Parameters:
  - X                  : Input matrix such that X.T @ BX = I
  - Z                  : Input matrix such that Z.T @ BZ = I
  - P                  : Input matrix such that P.T @ BP = I
  - AX, AZ, AP, BZ, BP : Additional input matrices
  - Lambda             : Optional vector of previous eigenvalues

  Returns:
  - hX         : Eigenvectors
  - Lambda[:m] : Eigenvalues
  """

  _, m = X.shape
  G1 = empty((3*m, 3*m))
  G2 = empty((3*m, 3*m))

  # G1[:] = [X, Z, P].T @ [AX, AZ, AP]
  if Lambda is None:
    G1[:m, :m] = matmul(X.T, AX)
  else:
    G1[:m, :m] = diag(Lambda)
  G1[:m, m:2*m] = matmul(X.T, AZ)
  G1[:m, 2*m:] = matmul(X.T, AP)
  G1[m:2*m, m:2*m] = matmul(Z.T, AZ)
  G1[m:2*m, 2*m:] = matmul(Z.T, AP)
  G1[2*m:, 2*m:] = matmul(P.T, AP)
  G1[m:2*m, :m] = G1[:m, m:2*m].T
  G1[2*m:, :m] = G1[:m, 2*m:].T
  G1[2*m:, m:2*m] = G1[m:2*m, 2*m:].T

  # G2[:] = [X, Z, P].T @ [BX, BZ, BP]
  G2[:m, :m] = eye(m)
  G2[m:2*m, m:2*m] = eye(m)
  G2[2*m:, 2*m:] = eye(m)
  G2[:m, m:2*m] = matmul(X.T, BZ)
  G2[:m, 2*m:] = matmul(X.T, BP)
  G2[m:2*m, 2*m:] = matmul(Z.T, BP)
  G2[m:2*m, :m] = G2[:m, m:2*m].T
  G2[2*m:, :m] = G2[:m, 2*m:].T
  G2[2*m:, m:2*m] = G2[m:2*m, 2*m:].T

  # Find L such that L @ L.T = G2
  L = cholesky(G2, lower=True)

  # G2[:] = L^(-1) @ G1 @ L^(-T)
  G2[:] = solve_triangular(L, G1, lower=True, overwrite_b=True)
  G2[:] = solve_triangular(L, G2.T, lower=True)
    
  # Solve eigenvalue problem
  Lambda, hY = eigh(G2)

  # hX = L^(-T) @ hY[:, :m]
  hX = solve_triangular(L, hY[:, :m], trans='T', lower=True)

  return hX, Lambda[:m]