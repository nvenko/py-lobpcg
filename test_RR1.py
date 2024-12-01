import numpy as np
from scipy.io import mmwrite, mmread
from RayleighRitz import RR1

np.random.seed(1)

m = 10 
n = 5 # n <= m

def get_X(m, n):
  return np.random.rand(m, n)

def get_A(m):
  A = np.random.rand(m, m)
  A += np.transpose(A)  
  return A @ A

X = get_X(m, n)
while np.linalg.matrix_rank(X) < n:
  X = get_X(n, m)

A = get_A(m)
while np.linalg.matrix_rank(A) < m:
  A = get_A(m)

B = get_A(m)
while np.linalg.matrix_rank(B) < m:
  B = get_A(m)

AX = A @ X; BX = B @ X
mmwrite("test_RR1_X.mtx", X)
mmwrite("test_RR1_AX.mtx", AX)
mmwrite("test_RR1_BX.mtx", BX)

X = mmread("test_RR1_X.mtx")
AX = mmread("test_RR1_AX.mtx")
BX = mmread("test_RR1_BX.mtx")

hX, Lambda = RR1(X, AX, BX)

err = np.linalg.norm(X.T @ AX @ hX - X.T @ BX @ hX @ np.diag(Lambda))
print(err) # 3.9791248551443245e-13 for m, n = 10, 5

mmwrite("test_RR1_hX.mtx", hX)
mmwrite("test_RR1_Lambda.mtx", np.diag(Lambda), symmetry='symmetric')


