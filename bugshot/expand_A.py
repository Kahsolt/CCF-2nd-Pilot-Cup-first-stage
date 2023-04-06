#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/05 

from P2 import *

print('[A]')
print(A)

A_ex = np.zeros([4, 4])
A_ex[:2, 2:] = A
A_ex[2:, :2] = A.conj().T

print('[A_ex]')
print(A_ex)

print('[A_ex]')
print(A_ex @ A_ex.conj().T)

eigen_A(A_ex)


print()
print('[invA]')
print(np.linalg.inv(A))

print('[invA?]')
vals, vecs = np.linalg.eig(A)
invA_hat = vecs @ np.diag(1 / vals) @ np.linalg.inv(vecs)
print(invA_hat)


print()
print('[invA]')
print(np.linalg.inv(A_ex))

print('[invA?]')
vals, vecs = np.linalg.eig(A_ex)
invA_hat = vecs @ np.diag(1 / vals) @ np.linalg.inv(vecs)
print(invA_hat)
