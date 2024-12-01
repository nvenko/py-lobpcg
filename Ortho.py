from scipy.linalg.blas import dgemm
from numpy import matmul
from scipy.linalg import eigh, norm
from numpy import empty, eye, finfo, diag, max, abs

def OrthoB(U, B, V, BU, norm_BV,
           tol_ortho=100*finfo(float).eps,
           tol_replace=10*finfo(float).eps,
           itmax1=6, itmax2=6):
  """
  Stathopoulos, A., & Wu, K. (2002)
  A block orthogonalization procedure with constant synchronization requirements
  SIAM Journal on Scientific Computing, 23(6), 2165-2182
  """
  
  _, p = BU.shape
  _, q = V.shape
  UtBU = empty((p, p))
  VtBU = empty((q, p))
  matmul(V.T, BU, out=VtBU)
  for _ in range(itmax1):
    U[:] = dgemm(alpha=-1.0, a=V, b=VtBU, c=U, overwrite_c=True, beta=1.0)
    BU[:] = B @ U
    for _ in range(itmax2):
      svqbB(U, BU, tol_replace)
      matmul(U.T, BU, out=UtBU)
      norm_U = norm(U)
      err = norm(UtBU - eye(p)) / (norm(BU) * norm_U)
      if err < tol_ortho:
        break
    matmul(V.T, BU, out=VtBU)
    err = norm(VtBU) / (norm_BV * norm_U)
    if err < tol_ortho:
      break

def svqbB(U, BU, tol):
  """
  Stathopoulos, A., & Wu, K. (2002)
  A block orthogonalization procedure with constant synchronization requirements
  SIAM Journal on Scientific Computing, 23(6), 2165-2182
  """

  _, p = U.shape
  UtBU = empty((p, p))
  matmul(U.T, BU, out=UtBU)
  D = diag(diag(UtBU) ** -0.5)
  matmul(D, matmul(UtBU, D), out=UtBU) # UtBU[:] = D @ UtBU @ D
  Theta, Z = eigh(UtBU)
  theta_abs_max = max(abs(Theta))
  Theta[Theta < tol * theta_abs_max] = tol * theta_abs_max
  # Z[:] = D @ Z @ diag(Theta ** -0.5)
  matmul(D, matmul(Z, diag(Theta ** -0.5)), out=Z)
  U[:] = matmul(U, Z) 
  BU[:] = matmul(BU, Z) 

def Ortho(U, V, 
          tol_ortho=100*finfo(float).eps, 
          tol_replace=10*finfo(float).eps, 
          itmax1=6, itmax2=6):
  """
  Stathopoulos, A., & Wu, K. (2002)
  A block orthogonalization procedure with constant synchronization requirements
  SIAM Journal on Scientific Computing, 23(6), 2165-2182
  """

  _, p = U.shape
  _, q = V.shape
  UtU = empty((p, p))
  VtU = empty((q, p))
  matmul(V.T, U, out=VtU)
  norm_V = norm(V)
  for _ in range(itmax1):
    U[:] = dgemm(alpha=-1.0, a=V, b=VtU, c=U, overwrite_c=True, beta=1.0)  
    for _ in range(itmax2):
      svqb(U, tol_replace)
      norm_U = norm(U)
      matmul(U.T, U, out=UtU)
      err = norm(UtU - eye(p)) / (norm_U ** 2)
      if err < tol_ortho:
        break
    matmul(V.T, U, out=VtU)
    err = norm(VtU) / (norm_V * norm_U)
    if err < tol_ortho:
      break
  
def svqb(U, tol):
  """
  Stathopoulos, A., & Wu, K. (2002)
  A block orthogonalization procedure with constant synchronization requirements
  SIAM Journal on Scientific Computing, 23(6), 2165-2182
  """

  _, p = U.shape
  UtU = empty((p, p))
  matmul(U.T, U, out=UtU)
  D = diag(diag(UtU) ** -0.5)
  matmul(D, matmul(UtU, D), out=UtU) # UtU[:] = D @ UtU @ D
  Theta, Z = eigh(UtU)
  theta_abs_max = max(abs(Theta))
  Theta[Theta < tol * theta_abs_max] = tol * theta_abs_max
  # Z[:] = D @ Z @ diag(Theta ** -0.5)
  matmul(D, matmul(Z, diag(Theta ** -0.5)), out=Z)
  U[:] = matmul(U, Z)