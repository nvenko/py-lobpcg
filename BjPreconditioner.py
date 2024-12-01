import numpy as np
from scipy.sparse.linalg import splu

def get_slice(i, nb, bsize, n):
  start = (i - 1) * bsize
  end = i * bsize if i < nb else n
  return slice(start, end)

class BJop:
  def __init__(self, n, nb, bsize, factos):
    self.n = n
    self.nb = nb
    self.bsize = bsize
    self.factos = factos

  def invT_array(self, R, Z):
    for i in range(1, self.nb + 1):
      sl = get_slice(i, self.nb, self.bsize, self.n)
      block = self.factos[i - 1]
      Z[sl, :] = block.solve(R[sl, :])

  def invT_vector(self, r, z):
    for i in range(1, self.nb + 1):
      sl = get_slice(i, self.nb, self.bsize, self.n)
      block = self.factos[i - 1]
      z[sl] = block.solve(r[sl])

  def invT(self, R):
    Z = R.copy()
    if len(Z.shape) == 2:
      self.invT_array(R, Z)
    else:
      self.invT_vector(R, Z)
    return Z
      
def assembleBj(nb, A):
  n = A.shape[0]
  bsize = n // nb
  factos = []
  for i in range(1, nb + 1):
    sl = get_slice(i, nb, bsize, n)
    block = A[sl, sl]
    factos.append(splu(block))
  return BJop(n, nb, bsize, factos)