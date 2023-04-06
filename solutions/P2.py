#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/03 

from typing import Tuple, List, Callable

from pyqpanda import *
import numpy as np
np.set_printoptions(suppress=True)
import scipy.linalg as spl

Matrix = np.ndarray
Vector = np.ndarray

DEBUG = True

# case of the special: `A12` is the original essay example, `b1` is the test case from Qiskit code
# NOTE: Aij is required to be hermitian (not needed to be unitary though)
r2 = np.sqrt(2)
A12 = np.asarray([    # eigval: 2, 1
  [3, 1],             # eigvec: [0.70710678, 0.70710678], [-0.70710678, 0.70710678]
  [1, 3],
]) / 2
A13 = np.asarray([    # eigval: 3, 1
  [2, 1],             # eigvec: [0.70710678, 0.70710678], [-0.70710678, 0.70710678]
  [1, 2],
])
A23 = np.asarray([    # eigval: 3, 2
  [5, 1],             # eigvec: [0.70710678, 0.70710678], [-0.70710678, 0.70710678]
  [1, 5],
]) / 2
b0 = np.asarray([1, 0])         # |b> = |0>, for basis test
b1 = np.asarray([0, 1])         # |b> = |1>
bp = np.asarray([1,  1]) / r2   # |b> = |+>
bn = np.asarray([1, -1]) / r2   # |b> = |->

# case of the target `question1()`
# svd(A) = P * D *Q
#   [-1 0][1    0][-1 -1]
#   [ 0 1][0 1/r2][ 1 -1]
A = np.asarray([     # eigval: 1.34463698, -1.05174376
  [   1,     1],     # eigvec: [0.9454285, 0.32582963], [-0.43812257, 0.89891524]
  [1/r2, -1/r2],
])
b = np.asarray([
    1/2, 
  -1/r2,
])

# tunable parameters of HHL_2x2 circuit
t0 = 2 * np.pi
r  = 4


def eigen_A(A:Matrix, title='A'):
  ''' sanity check A = P*D*P^(-1) and Ax = λx '''

  eigenvals, eigenvecs = np.linalg.eig(A)
  assert np.allclose(A, eigenvecs @ np.diag(eigenvals) @ np.linalg.inv(eigenvecs))

  print(f'eigen({title}):')
  for i, lbd in enumerate(eigenvals):
    x = eigenvecs[:, i]                 # eigenvec is the column vector
    assert np.allclose(A @ x, lbd * x)

    if DEBUG and 'A eigen pairs':
      print(f'  λ{i}: {lbd}')
      print(f'  v{i}: {x}')

def q0q1q2q3_to_q3(qstate:List[complex]) -> List[complex]:
  ''' squeeze |q3> from |q0,q1,q2,q3>; should use rank-1 decompose, but we play tricks here '''

  if len(qstate) == 2:    return qstate
  if len(qstate) == 2**4: return [qstate[0], qstate[2**(4-1)]]
  raise ValueError


def transform(A:Matrix, b:Vector, Ar:Matrix=A12) -> Tuple[Tuple[Matrix, Vector], Tuple[Matrix, float]]:
  '''
    Transforming an arbitray Ax = b to form of Ar * y = b_n:
                A    * x = b
            (Ar * D) * x = b          # left-decompose A by a known matrix Ar, this is NOT effecient in any sense though
      Ar * (D / |b| * x) = b / |b|    # normalize right side b, turining `b/b_n` a unit vector
                  Ar * y = b_n        # rename, suits the form required by HHL_2x2 circuit
  '''

  D = np.linalg.inv(Ar) @ A
  b_norm = np.linalg.norm(b)
  b_n = b / b_norm

  # the validated A and b for new equation, and stats needed for inversion 
  return (Ar, b_n), (D, b_norm)

def transform_inv(y:Vector, D:Matrix, b_norm:float) -> Vector:
  '''
    Solve x from y = D / |b| * x, this is NOT effecient in any sense though:
      D / |b| * x = y
                x = (D / |b|)^(-1) * y
                x = D^(-1) * |b| * y
  '''

  return np.linalg.inv(D) @ (b_norm * y)


def encode_b(b:Vector, q:Qubit) -> QCircuit:
  '''
    Amplitude encode a unit vector b = [b0, b1] to |b> = b0|0> + b1|1> via the universal rotaion gate U3:
                                          U(theta, phi, lmbd) |0> = |b>
      [          cos(theta/2), -e^(i*lmbd)      *sin(theta/2)][1] = [b0]
      [e^(i*phi)*sin(theta/2),  e^(i*(lmbd+phi))*cos(theta/2)][0] = [b1]
    
    NOTE: single RY gate does not work, it lacks the coeff `e^(i*phi)` compared with U3 gate
                         RY(theta) |0> = |b>
      [cos(theta/2), -sin(theta/2)][1] = [b0]
      [sin(theta/2),  cos(theta/2)][0] = [b1]
  '''

  if isinstance(b, list): b = np.asarray(b)
  assert isinstance(b, Vector), 'should be a Vector/np.ndarray'
  assert len(b.shape) == 1 and b.shape[0] == 2, 'should be a vector lengthed 2'
  b = b.astype(np.complex128)
  assert (np.linalg.norm(b) - 1.0) < 1e-5, 'should be a unit vector'

  theta = 2 * np.arccos(b[0])
  phi   = np.log(b[1] / np.sin(theta / 2)) / 1j
  lmbd  = 0
  cq = QCircuit() << U3(q, theta.real, phi.real, lmbd)

  if DEBUG and '|b> = b0|0> + b1|1> amplitude':
    print('b:', b)
    qvm.directly_run(QProg() << cq)
    qstate = qvm.get_qstate()   # need amp other than prob
    print('|b>:', q0q1q2q3_to_q3(qstate))

  return cq

def encode_A(A:Matrix, q:Qubit, theta:float=t0) -> QGate:
  ''' Unitary generator encode hermitian A to a unitary U = exp(iAθ), where conventionally θ = t0 / t '''

  u = spl.expm(1j* A * theta).astype(np.complex128)   # FIXME: explain why this is a valid unitary??
  assert np.allclose(u @ u.conj().T, np.eye(len(u)))

  if DEBUG and 'U = exp(iAθ) unitarity':
    eigen_A(A)
    print(f'U = exp(iAθ), theta={theta}:')
    print(u)
    eigen_A(u, 'u')     # eigval of a unitary holds that |λi|=1

  # FIXME: this should be where errors mainly come from :(
  return matrix_decompose(q, u, DecompositionMode.QR)


def HHL_2x2_cao13(A:Matrix=A12, b:Vector=b1) -> QCircuit:
  '''
    Implementation of the toy HHL circuit solving a minimal 2x2 system using only 4 qubits
    given in essay "Quantum Circuit Design for Solving Linear Systems of Equations" by Yudong Cao, et al.
      - https://arxiv.org/abs/1110.2232
  '''

  assert any([A is someA for someA in [A12, A13, A23]]), 'A should be chosen from [A12, A13, A23]'
  assert (np.linalg.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

  # q0: perform RY to inverse λi
  # q1~q2: QPE scatter A => Σ|λi>, λi is integer approx to eigvals of A
  # q3: amplitude encode b => |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  # Step 1: prepare |b>, very toyish :(
  enc_b = encode_b(b, q3)

  # Step 2: QPE of A over |b>
  # notes for QPE part:
  # θ=t0/4, exp(i*A12*(pi/2))   // encode_A(A12, q3, theta=t0/4)
  #  [-1+1j -1-j]
  #  [-1-1j -1+j] / 2
  # θ=t0/2, exp(i*A12*(pi))     // encode_A(A12, q3, theta=t0/2)
  #  [0, 1]
  #  [1, 0]
  # notes for iQFT part:
  #  - 2-qubit QFT: (H @ I) * Control(R2) * (I @ H), iQFT is the dagger
  #  - R2 = S = Z^(1/2) = P(pi/2) = R(pi/2), they are numerically all the same!!!
  qpe = QCircuit() \
      << H([q1, q2]) \
      << encode_A(A, q3, theta=t0/4).control(q2) \
      << encode_A(A, q3, theta=t0/2).control(q1) \
      << SWAP(q1, q2) \
      << H(q2) \
      << S(q2).dagger().control(q1) \
      << H(q1) \
      << SWAP(q1, q2)

  # Step 3: controled-Y rotate θj = 2*arcsin(C/λi) ≈ 2*C/λi; 2^(1-r)*pi = 2*C
  # r is a tunable hyper parameter between log2(2*pi) ~ log2(2*pi/w), where w is the minimal angle resolution
  # when w=2*pi/360, r is in range [2.651496129472319, 8.491853096329674]
  r = 4                   # following the essay
  rc = QCircuit() \
     << RY(q0, 2*np.pi/2**r).control(q1) \
     << RY(q0,   np.pi/2**r).control(q2)

  return enc_b << qpe << rc << qpe.dagger()

def HHL_2x2_cao13_qiskit(A:Matrix=A12, b:Vector=b1) -> QCircuit:
  '''
    Same as `HHL_2x2_cao13` but translated from a Qiskit implementation
      - https://blog.csdn.net/m0_37622530/article/details/87938105
  '''

  assert any([A is someA for someA in [A12, A13]]), 'A should be chosen from [A12, A13]'
  assert (np.linalg.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

  # q0: perform RY to inverse λi
  # q1~q2: QPE scatter A => Σ|λi>, λi is integer approx to eigvals of A
  # q3: amplitude encode b => |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  # Step 1: prepare |b>, very toyish :(
  enc_b = encode_b(b, q3)

  # Step 2: QPE of A over |b>
  # u2 = [1, 0,     // U1(q2, np.pi/2)
  #       0, i]
  # u1 = [1, 0,     // U1(q1, np.pi)
  #       0, -1]
  qpe = QCircuit() \
      << H(q1) \
      << H(q2) \
      << U1(q2, np.pi/2) \
      << U1(q1, np.pi) \
      << CNOT(q3, q2) \
      << SWAP(q1, q2) \
      << H(q2) \
      << U1(q2, -np.pi/2).control(q1) \
      << H(q1) \
      << (SWAP(q1, q2) if A is A12 else X(q1))

  # Step 3: controled-Y rotate θj = 2*arcsin(C/λi) ≈ 2*C/λi; 2^(1-r)*pi = 2*C
  # r1 = [0.9807852804032304, -0.19509032201612825,     // U3(q0, np.pi/8,  0, 0)
  #       0.19509032201612825, 0.9807852804032304]
  # r2 = [0.9951847266721969, -0.0980171403295606,      // U3(q0, np.pi/16, 0, 0)
  #       0.0980171403295606, 0.9951847266721969]
  rc = QCircuit() \
     << U3(q0, np.pi/8,  0, 0).control(q1) \
     << U3(q0, np.pi/16, 0, 0).control(q2)

  return enc_b << qpe << rc << qpe.dagger()

def HHL_2x2_cao13_qiskit_corrected(A:Matrix=A12, b:Vector=b1) -> QCircuit:
  '''
    Same as `HHL_2x2_cao13_qiskit` but corrected errors in QPE part I think...
  '''

  assert any([A is someA for someA in [A12, A13]]), 'A should be chosen from [A12, A13]'
  assert (np.linalg.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

  # q0: perform RY to inverse λi
  # q1~q2: QPE scatter A => Σ|λi>, λi is integer approx to eigvals of A
  # q3: amplitude encode b => |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  # Step 1: prepare |b>, very toyish :(
  enc_b = encode_b(b, q3)

  # Step 2: QPE of A over |b>
  qpe = QCircuit() \
      << H(q1) \
      << H(q2) \
      << U1(q3, np.pi/2).control(q2) \
      << U1(q3, np.pi)  .control(q1) \
      << SWAP(q1, q2) \
      << H(q2) \
      << U1(q2, -np.pi/2).control(q1) \
      << H(q1) \
      << (SWAP(q1, q2) if A is A12 else X(q1))

  # Step 3: controled-Y rotate θj = 2*arcsin(C/λi) ≈ 2*C/λi; 2^(1-r)*pi = 2*C
  rc = QCircuit() \
     << U3(q0, np.pi/8,  0, 0).control(q1) \
     << U3(q0, np.pi/16, 0, 0).control(q2)

  return enc_b << qpe << rc << qpe.dagger()

def HHL_2x2_pan13(A:Matrix=A12, b:Vector=b1) -> QCircuit:
  ''' 
    Implementation of the toy HHL circuit solving a minimal 2x2 system using only 4 qubits
    given in essay "Experimental realization of quantum algorithm for solving linear systems of equations" by Pan et al.
      - https://arxiv.org/abs/1302.1946
  '''

  assert A is A12, 'A must be A12'
  assert (np.linalg.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

  # q0: perform RY to inverse λi
  # q1~q2: QPE scatter A => Σ|λi>, λi is integer approx to eigvals of A
  # q3: amplitude encode b => |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  # Step 1: prepare |b>, very toyish :(
  enc_b = encode_b(b, q3)

  # Step 2: QPE of A over |b>
  qpe = QCircuit()
  qpe << H([q1, q2]) \
      << encode_A(A, q3, theta=-t0/4).control(q1) \
      << encode_A(A, q3, theta=-t0/2).control(q2) \
      << H(q2) \
      << S(q2).control(q1) \
      << H(q1)

  # Step 3: R(λ^(-1)) rotation
  r = 2                         # following the essay
  rc = QCircuit() \
     << SWAP(q1, q2) \
     << RY(q0,   np.pi/2**r).control(q1) \
     << RY(q0, 2*np.pi/2**r).control(q2)

  return enc_b << qpe << rc << qpe.dagger()

def HHL_2x2_cai13(A:Matrix=A12, b:Vector=bp, init:bool=False) -> QCircuit:
  '''
    Highly optimized 2x2 HHL circuit in experiments of real optical quantum systems
    given in essay "Experimental quantum computing to solve systems of linear equations" by Cai et al.
      - https://arxiv.org/abs/1302.4310
  '''

  assert A is A12, 'A must be A12'
  assert any([np.allclose(b, someB, 1e-3, 1e-3) for someB in[bp, bn, b0]]), 'b must chosen from [bp, bn, b0]'

  def H_theta(q:Qubit, theta:float) -> QGate:
    ''' The H(theta) gate defined in the essay is really a mystic magic :( '''
    mat = np.asarray([
      [np.cos(2*theta),  np.sin(2*theta)],
      [np.sin(2*theta), -np.cos(2*theta)],
    ])
    return QOracle([q], mat)

  # q0: Ancilla |0>A
  # q1~q2: Register |0>R1/|0>R2
  # q3: Input |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  qc = QCircuit()
  if init: qc << H([q0, q1, q2, q3])    # GHZ-state init following the essay
  if   np.allclose(b, b0, 1e-3, 1e-3): qc << I(q3)
  elif np.allclose(b, bp, 1e-3, 1e-3): qc << H(q3)
  elif np.allclose(b, bn, 1e-3, 1e-3): qc << X(q3) << H(q3)

  qc << CNOT(q3, q2) \
     << CNOT(q2, q1) \
     << H(q3) \
     << X(q2) \
     << H_theta(q0, np.pi/ 8).control(q1) \
     << H_theta(q0, np.pi/16).control(q2) \
     << H(q2) \
     << H(q1)

  return qc

def HHL_2x2_qpanda(A:Matrix=A12, b:Vector=bp, ver:int='QPanda') -> QCircuit:
  '''
    The implementations from QPanda:
      - #ba0b08f4fb86a955bee2607045bd5130523e604c init~       toy implementation accepting no inputs, maybe solving `A12 x = b0`
      - #509020cfae8eb1d0299f691bf52492c727083d18 29/09/2020  general purposed implementation, move to `QAlg/HHL/HHL.cpp`, works fine except wrong sign
      - #3f41c92cdd8a138d01e350034b22d4908ecbbfec 17/06/2021  add parameter `precision_cnt` for `HHL_solve_linear_equations()`
      - #acc794aec4118a19b7127a4448460faf0eb9219f 26/07/2021  implementation code moved to `Extensions`, added to `.gitignore` and removed from track
    The toy implementation is also known as QRunes:
      - https://qrunes-tutorial.readthedocs.io/en/latest/chapters/algorithms/HHL_Algorithm.html
      - https://github.com/OriginQ/QRunes/blob/master/source/chapters/algorithms/HHL_Algorithm.rst
  '''

  assert ver in ['QRunes-code', 'QRunes-graph', 'QPanda']

  # q0: rotate
  # q1~q2: QPE
  # q3: ampl_enc |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  if ver == 'QRunes-code':    # this is impl of C++ code
    enc_b = QCircuit() \
          << RY(q3, np.pi/2)

    pse = QCircuit() \
        << H(q1) \
        << H(q2) \
        << RZ(q2, 0.75*np.pi) \
        << CU(np.pi, 1.5*np.pi, -np.pi/2, np.pi/2, q2, q3) \
        << RZ(q1, 1.5*np.pi) \
        << CU(np.pi, 1.5*np.pi,  np.pi,   np.pi/2, q1, q3) \
        << CNOT(q1, q2) \
        << CNOT(q2, q1) \
        << CNOT(q1, q2) \
        << H(q2) \
        << CU(-np.pi/4, -np.pi/2, 0, 0, q2, q1) \
        << H(q1)

    cr = QCircuit() \
      << X(q1) \
      << RY(q0, np.pi).control([q1, q2]) \
      << X(q1) \
      << X(q2) \
      << RY(q0, np.pi / 3).control([q1, q2]) \
      << X(q2) \
      << RY(q0, 0.679673818908).control([q1, q2])   # arcsin(1/3)

    cq = enc_b << pse << cr

  if ver == 'QRunes-graph':   # this is impl of the shown graph
    pse = QCircuit() \
        << H(q1) \
        << H(q2) \
        << RZ(q2, 0.75*np.pi) \
        << RX(q3, -np.pi/2).control(q2) \
        << RZ(q1, 1.5*np.pi) \
        << RX(q3, -np.pi).control(q1) \
        << SWAP(q1, q2) \
        << H(q2) \
        << RZ(q1, -np.pi/2).control(q2) \
        << H(q1)

    cr = QCircuit() \
      << X(q1) \
      << RY(q0, np.pi).control([q1, q2]) \
      << X(q1) \
      << X(q2) \
      << RY(q0, np.pi / 3).control([q1, q2]) \
      << X(q2) \
      << RY(q0, 0.679673818908).control(q2)   # arcsin(1/3)

    cq = pse << cr << pse.dagger()

  if ver == 'QPanda':
    enc_b = encode_b(b, q3)

    qpe = QCircuit() \
        << H(q1) \
        << H(q2) \
        << matrix_decompose(q3, spl.expm(1j*A*2*np.pi/1)).control(q2) \
        << matrix_decompose(q3, spl.expm(1j*A*2*np.pi/2)).control(q1) \
        << H(q2) \
        << CR(q1, q2, 2*np.pi/2) \
        << H(q1)

    cr = QCircuit() \
       << RY(q0, 2*np.arcsin(-1/3)).control([q1, q2]) \
       << RY(q0, 2*np.arcsin(-1/2)).control([q1, q2]) \
       << X(q1) << X(q2) \
       << RY(q0, 2*np.arcsin(-1/1)).control([q1, q2]) \
       << X(q1)

    cq = enc_b << qpe << cr << qpe.dagger()

  return cq

def HHL_2x2_test(A:Matrix=A12, b:Vector=b1) -> QCircuit:
  ''' My modified version, looking for best param '''

  assert any([A is someA for someA in [A12, A13, A23]]), 'A should be chosen from [A12, A13, A23]'
  assert (np.linalg.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

  # q0: perform RY to inverse λi
  # q1~q2: QPE scatter A => Σ|λi>, λi is integer approx to eigvals of A
  # q3: amplitude encode b => |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  # Step 1: prepare |b>, very toyish :(
  enc_b = encode_b(b, q3)

  # Step 2: QPE of A over |b>
  qpe = QCircuit() \
      << H([q1, q2]) \
      << encode_A(A, q3, theta=t0/4).control(q2) \
      << encode_A(A, q3, theta=t0/2).control(q1) \
      << H(q2) \
      << S(q2).dagger().control(q1) \
      << H(q1)

  # Step 3: controled-Y rotate θj = 2*arcsin(C/λi) ≈ 2*C/λi; 2^(1-r)*pi = 2*C
  r = 4                   # following the essay
  rc = QCircuit() \
     << RY(q0, 2*np.pi/2**r).control(q1) \
     << RY(q0,   np.pi/2**r).control(q2)

  return enc_b << qpe << rc << qpe.dagger()


def HHL(A:Matrix, b:Vector, HHL_cq:Callable) -> Tuple[List[float], str]:
  '''
    Solving linear system equation `Ax = b` by quantum simulation in poly(logN) steps
      - https://arxiv.org/abs/0811.3171
      - https://arxiv.org/abs/1110.2232
      - https://arxiv.org/abs/1302.1210
      - https://arxiv.org/abs/1805.10549
      - https://arxiv.org/abs/1302.4310
      - https://arxiv.org/abs/1302.1946
      - https://en.wikipedia.org/wiki/Quantum_algorithm_for_linear_systems_of_equations
      - https://zhuanlan.zhihu.com/p/164375189
      - https://zhuanlan.zhihu.com/p/426811646
      - https://www.qtumist.com/post/5212
      - https://pyqpanda-toturial.readthedocs.io/zh/latest/HHL.html
    NOTE: 
      - Input `A` and `b` for this function are arbitary, they will be auto-transformed for valid quantum process
      - Due to resource limit of 4 qubits and explorations of our ancestors, we do NOT solve arbitary linear equation all throught quantum process; 
        but only solve one special prototype case in quantum manner, then **reduce** any other cases to it in a classical way :(
        This special case is artificially designed to allow eigenvals of matrix A is stored without precision loss
        i.e. we choose a unitary A with eigen values of [1, 2], [1, 3] or [2, 3], so that eigenvals can be exactly stored using 2-qubits stated |01>, |10> or |11>
  '''
  assert isinstance(A, Matrix) and tuple(A.shape) == (2, 2)
  assert isinstance(b, Vector) and tuple(b.shape) == (2,)

  # Step 1: transform in a classical manner
  (Ar, b_n), (D, b_norm) = transform(A, b, Ar=A12)

  if DEBUG and 'transform':
    print('Ar:') ; print(Ar)
    print('b_n:', b_n)
    print('D:')  ; print(D)
    print('b_norm:', b_norm)

    x_hat = np.linalg.solve(A, b)       # direct true answer
    y_hat = np.linalg.solve(Ar, b_n)    # true answer of transformed equation

  # Step 2: solving Ar * y = b_n in a quantum manner
  global qvm
  qvm = CPUQVM()
  qvm.init_qvm()

  hhl_cq = HHL_cq(Ar, b_n)

  if DEBUG and 'HHL_2x2 circuit':
    print(hhl_cq)

  prog = QProg() << hhl_cq
  qvm.directly_run(prog)
  y = q0q1q2q3_to_q3(qvm.get_qstate())

  try:    ircode = to_originir(prog, qvm)
  except: ircode = ''
  #qvm.qFree_all(qvm.get_allocate_qubits())   # FIXME: buggy, stucks to SEGV
  qvm.finalize()

  # Step 3: inv-transform in a classical manner
  y = np.asarray(y, dtype=np.complex128)
  x = transform_inv(y, D, b_norm)
  x = x.real.tolist()

  if DEBUG and 'solution':
    print('x_hat:', x_hat)
    print('y_hat:', y_hat)
    print('y:', y)
    print('x:', x)

  return x, ircode


def question1() -> Tuple[list, str]:
  return HHL(A, b, HHL_2x2_cao13)


def go(A:Matrix, b:Vector, HHL_cq:Callable=HHL_2x2_cao13, prt_ircode=False):
  try:
    z = np.linalg.solve(A, b)
  except np.linalg.LinAlgError:   # ignore bad rand case if singular
    return

  x, ircode = HHL(A, b, HHL_cq)
  print('  solution:', x)
  print('  correct:', z)

  if prt_ircode: print(ircode)
  print()


if __name__ == '__main__':
  if not 'sanity check encode_b()':
    q = qvm.qAlloc()
    for _ in range(4):
      b = np.random.uniform(size=[2], low=-1.0, high=1.0)
      b /= np.linalg.norm(b)
      encode_b(b, q)
    # qvm.qFree(q)    # FIXME: buggy, stucks to SEGV

  if not 'sanity check eigen_A()':
    for A in [A12, A13, A23, A]:
      eigen_A(A)

  # solve the specified target linear equation
  print('[target question] (different methods)')
  print('>> HHL_2x2_test')
  go(A, b, HHL_2x2_test, prt_ircode=False)
  print('>> HHL_2x2_cao13')
  go(A, b, HHL_2x2_cao13)
  print('>> HHL_2x2_cao13_qiskit')
  go(A, b, HHL_2x2_cao13_qiskit)
  print('>> HHL_2x2_cao13_qiskit_corrected')
  go(A, b, HHL_2x2_cao13_qiskit_corrected)

  HHL_cq = HHL_2x2_test
  DEBUG = False

  # solve the toy example in essay
  print('[essay question]')
  go(A12, b1, HHL_cq)

  # solve random linear equations
  print('[random questions]')
  for _ in range(10):
    A = np.random.uniform(size=[2, 2], low=-1.0, high=1.0)
    b = np.random.uniform(size=[2],    low=-1.0, high=1.0)
    go(A, b, HHL_cq)
