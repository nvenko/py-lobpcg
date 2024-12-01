from LobpcgGeneralized import *
from LobpcgStandard import *

def LOBPCG(A, X0, nev,
           B=None, T=None, itmax=200, tol=1e-6,
           method='Skip_ortho',
           A_products='implicit',
           B_products='implicit'):
  """
  LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) method.
    
  Parameters:
  A          : left  hand-side operator, symmetric positive definite, n-by-n
  X0         : initial iterates, n-by-m (m < n)
  nev        : number of eigenvalues to compute.
  B          : right hand-side operator, symmetric positive definite, n-by-n
  T          : precondontioner, symmetric positive definite, n-by-n
  itmax      : maximum number of iterations
  tol        : tolerance used for convergence criterion
  method     : type of LOBPCG iterations among ('Basic', 'BLOPEX', 'Ortho', 'Skip_ortho')
  A_products : if "implicit", the matrix products with A are updated implicitly
  B_products : if "implicit", the matrix products with B are updated implicitly
  
  Returns:
  Lambda : last iterates of least dominant eigenvalues, m-by-1
  X      : last iterates of least dominant eigenvectors, n-by-m
  res    : normalized norms of eigenresiduals, m-by-it
  """
    
  if B is not None:
    if method == 'Basic':
      Lambda, X, res = Basic_LOBPCG_gen(A, B, X0, nev,
                                        T=T, itmax=itmax, tol=tol,
                                        A_products=A_products,
                                        B_products=B_products)
    elif method == 'BLOPEX':
      Lambda, X, res = BLOPEX_LOBPCG_gen(A, B, X0, nev,
                                         T=T, itmax=itmax, tol=tol,
                                         A_products=A_products,
                                         B_products=B_products)
    elif method == 'Ortho':
      Lambda, X, res = Ortho_LOBPCG_gen(A, B, X0, nev,
                                        T=T, itmax=itmax, tol=tol,
                                        A_products=A_products,
                                        B_products=B_products)
    elif method == 'Skip_ortho':
      Lambda, X, res = Skip_ortho_LOBPCG_gen(A, B, X0, nev,
                                             T=T, itmax=itmax, tol=tol,
                                             A_products=A_products,
                                             B_products=B_products)
  else:
    if method == 'Basic':
      Lambda, X, res = Basic_LOBPCG_standard(A, X0, nev,
                                             T=T, itmax=itmax, tol=tol,
                                             A_products=A_products)
    elif method == 'BLOPEX':
      Lambda, X, res = BLOPEX_LOBPCG_standard(A, X0, nev,
                                              T=T, itmax=itmax, tol=tol,
                                              A_products=A_products)
    elif method == 'Ortho':
      Lambda, X, res = Ortho_LOBPCG_standard(A, X0, nev,
                                             T=T, itmax=itmax, tol=tol,
                                             A_products=A_products)
    elif method == 'Skip_ortho':
      Lambda, X, res = Skip_ortho_LOBPCG_standard(A, X0, nev,
                                                  T=T, itmax=itmax, tol=tol,
                                                  A_products=A_products)
      
  return Lambda, X, res