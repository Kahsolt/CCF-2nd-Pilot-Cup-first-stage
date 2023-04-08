#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/03 

from typing import Tuple, List, Callable, Union

from pyqpanda import *
import numpy as np
np.set_printoptions(suppress=True)
import numpy.linalg as npl
import scipy.linalg as spl

Matrix = np.ndarray
Vector = np.ndarray

CHECK = False     # force check safe input of each circuit
DEBUG = True

# case of the special: `A12` is the original essay example, `b1` is the test case from Qiskit code
# NOTE: Aij is required to be hermitian (not needed to be unitary though)
r2 = np.sqrt(2)
pi = np.pi
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
Ah = np.asarray([     # eigval: 3, 2
  [3, -1],             # eigvec: [0.70710678, 0.70710678], [-0.70710678, 0.70710678]
  [-1, 3],
]) / 3
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
t0 = 2 * pi
r  = 4


def eigen_A(A:Matrix, title='A'):
  ''' sanity check A = P*D*P^(-1) and Ax = λx '''

  eigenvals, eigenvecs = npl.eig(A)
  assert np.allclose(A, eigenvecs @ np.diag(eigenvals) @ npl.inv(eigenvecs))

  print(f'eigen({title}):')
  for i, lbd in enumerate(eigenvals):
    x = eigenvecs[:, i]                 # eigenvec is the column vector
    assert np.allclose(A @ x, lbd * x)

    if DEBUG and 'A eigen pairs':
      print(f'  λ{i}: {lbd}')
      print(f'  v{i}: {x}')

def project_q3(qstate:List[complex], is_first:bool=False) -> List[complex]:
  ''' gather all cases of ancilla qubit to be |1>, then pick out the target |q> only '''

  #print(np.round(qstate, 4).real)
  if is_first:
    # the first |q0> is the target qubit
    return qstate[len(qstate)//2:][:2]
  else:
    # the last |q3> is the target qubit
    return [qstate[0], qstate[len(qstate)//2]]


def transform(A:Matrix, b:Vector, meth:str='hermitian', A_r:Matrix=A12) -> Tuple[Tuple[Matrix, Vector], Tuple[Matrix, float]]:
  ''' Transforming an arbitray Ax = b to equivalent forms '''

  assert meth in ['nothing', 'reduce', 'hermitian']

  if meth == 'nothing':
    return (A, b), (None, None)

  if meth == 'reduce':
    '''
      Reduce arbitray Ax = b to Ar * y = b_n where Ar is hand-crafted well-known and b_n is unit:
                  A  * x = b
            (Ar * D) * x = b          # left-decompose A by a known matrix Ar, this is NOT effecient in any sense though
      Ar * (D / |b| * x) = b / |b|    # normalize right side b, turining it a unit vector
                  Ar * y = b_n        # rename, suits the form required by HHL_2x2 circuit
    '''
    D = npl.inv(A_r) @ A
    b_norm = npl.norm(b)
    b_n = b / b_norm

    # the validated A and b for new equation, and stats needed for inversion 
    return (A_r, b_n), (D, b_norm)

  if meth == 'hermitian':
    '''
      Multiply a certain B on both side to make A a hermitian, then :
                         A  * x = b
                   (A' * A) * x = A' * b                 # (if necessary) multiply by A.dagger at both side to assure A_hat is hermitian
      (A' * A) * (x / |A' * b|) = (A' * b) / |A' * b|    # normalize right side A' * b, turining it a unit vector
                         Ah * y = b_n                    # rename, suits the form required by HHL_2x2 circuit
    '''

    if not np.allclose(A, A.conj().T):    # if A is not hermitian, make it hermitian
      B = A.conj().T                      # A' * A is promised to be hermitian in math
      A_h = B @ A
      b   = B @ b
      assert np.allclose(A_h, A_h.conj().T)
    else:
      A_h = A

    b_norm = npl.norm(b)
    b_n = b / b_norm

    # the validated A and b for new equation, and stats needed for inversion 
    return (A_h, b_n), (None, b_norm)

def transform_inv(y:Vector, D:Matrix, b_norm:float, meth:str='hermitian') -> Vector:
  ''' Solve x from transformed answer y '''

  assert meth in ['nothing', 'reduce', 'hermitian']

  if meth == 'nothing':
    return y

  if meth == 'reduce':
    '''
      Solve x from y = D / |b| * x, this is NOT effecient in any sense though:
        D / |b| * x = y
                  x = (D / |b|)^(-1) * y
                  x = D^(-1) * |b| * y
    '''

    return npl.inv(D) @ (b_norm * y)

  if meth == 'hermitian':
    '''
      Solve x from y = x / |A' * b|:
        x / |A' * b| = y
                   x = |A' * b| * y
    '''

    return y if abs(1.0 - b_norm) < 1e-5 else b_norm * y


def encode_b(b:Vector, q:Qubit, gate:QGate=RY) -> QCircuit:
  '''
    Amplitude encode a unit vector b = [b0, b1] to |b> = b0|0> + b1|1> via the universal rotaion gate U3:
                                          U(theta, phi, lmbd) |0> = |b>
      [          cos(theta/2), -e^(i*lmbd)      *sin(theta/2)][1] = [b0]
      [e^(i*phi)*sin(theta/2),  e^(i*(lmbd+phi))*cos(theta/2)][0] = [b1]
    or use RY+Z gate, to fix the lacking the coeff `e^(i*phi)` compared with U3 gate when b1 < 0
                         RY(theta) |0> = |b>
      [cos(theta/2), -sin(theta/2)][1] = [b0]
      [sin(theta/2),  cos(theta/2)][0] = [b1]
  '''
  if isinstance(b, list): b = np.asarray(b)
  assert isinstance(b, Vector), 'should be a Vector/np.ndarray'
  assert tuple(b.shape) == (2,), 'should be a vector lengthed 2'
  assert (npl.norm(b) - 1.0) < 1e-5, 'should be a unit vector'
  assert gate in [RY, U3]

  if gate is RY:
    theta = 2 * np.arccos(b[0])
    cq = QCircuit() << RY(q, theta)
    if b[1] < 0: cq << Z(q)             # fix sign of |1> part

  if gate is U3:
    theta = 2 * np.arccos(b[0])
    phi = np.log(b[1] / np.sin(theta / 2)) / 1j
    lmbd  = 0
    cq = QCircuit() << U3(q, theta, phi, lmbd)

  if DEBUG and '|b> = b0|0> + b1|1> amplitude':
    print('b:', b)
    qvm.directly_run(QProg() << cq)
    print('|b>:', project_q3(qvm.get_qstate()))   # need amp other than prob

  return cq

def encode_A(A:Matrix, q:Qubit, theta:float=t0, k:float=0, encoder=QOracle) -> Union[QGate, QCircuit]:
  ''' Unitary generator encode hermitian A to a unitary up to power of 2: U^2^k = exp(iAθ)^2^k '''
  assert isinstance(A, Matrix), 'should be a Matrix/np.ndarray'
  assert tuple(A.shape) == (2, 2), 'should be a 2x2 matrix'
  assert np.allclose(A, A.conj().T), 'should be a hermitian matrix'
  assert encoder in [QOracle, matrix_decompose]

  u = spl.expm(1j* A * theta).astype(np.complex128)   # FIXME: explain why this is a valid unitary??
  assert np.allclose(u @ u.conj().T, np.eye(len(u)))

  if DEBUG and 'U = exp(iAθ) unitarity':
    eigen_A(A)
    print(f'U = exp(iAθ), theta={theta}:')
    print(u)
    # eigval of u holds that |λi|=1, eigvec are the same as A
    # in fact: exp(iAt) = Σj exp(i*λj*t) |vj><vj|, only exp() over the eigenvals
    eigen_A(u, 'u')

  return encoder(q, npl.matrix_power(u, 2**k))


def HHL_2x2_cao13(A:Matrix=A12, b:Vector=b1) -> QCircuit:
  '''
    Implementation of the toy HHL circuit solving a minimal 2x2 system using only 4 qubits
    given in essay "Quantum Circuit Design for Solving Linear Systems of Equations" by Yudong Cao, et al.
      - https://arxiv.org/abs/1110.2232
  '''

  if CHECK:
    assert any([A is someA for someA in [A12, A13, A23]]), 'A should be chosen from [A12, A13, A23]'
    assert (npl.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

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
     << RY(q0, 2*pi/2**r).control(q1) \
     << RY(q0,   pi/2**r).control(q2)

  return enc_b << qpe << rc << qpe.dagger()

def HHL_2x2_cao13_qiskit(A:Matrix=A12, b:Vector=b1) -> QCircuit:
  '''
    Same as `HHL_2x2_cao13` but translated from a Qiskit implementation
      - https://blog.csdn.net/m0_37622530/article/details/87938105
  '''

  if CHECK:
    assert any([A is someA for someA in [A12, A13]]), 'A should be chosen from [A12, A13]'
    assert (npl.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

  # q0: perform RY to inverse λi
  # q1~q2: QPE scatter A => Σ|λi>, λi is integer approx to eigvals of A
  # q3: amplitude encode b => |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  # Step 1: prepare |b>, very toyish :(
  enc_b = encode_b(b, q3)

  # Step 2: QPE of A over |b>
  # u2 = [1, 0,     // U1(q2, pi/2)
  #       0, i]
  # u1 = [1, 0,     // U1(q1, pi)
  #       0, -1]
  qpe = QCircuit() \
      << H(q1) \
      << H(q2) \
      << U1(q2, pi/2) \
      << U1(q1, pi) \
      << CNOT(q3, q2) \
      << SWAP(q1, q2) \
      << H(q2) \
      << U1(q2, -pi/2).control(q1) \
      << H(q1) \
      << (SWAP(q1, q2) if A is A12 else X(q1))

  # Step 3: controled-Y rotate θj = 2*arcsin(C/λi) ≈ 2*C/λi; 2^(1-r)*pi = 2*C
  # r1 = [0.9807852804032304, -0.19509032201612825,     // U3(q0, pi/8,  0, 0)
  #       0.19509032201612825, 0.9807852804032304]
  # r2 = [0.9951847266721969, -0.0980171403295606,      // U3(q0, pi/16, 0, 0)
  #       0.0980171403295606, 0.9951847266721969]
  rc = QCircuit() \
     << U3(q0, pi/8,  0, 0).control(q1) \
     << U3(q0, pi/16, 0, 0).control(q2)

  return enc_b << qpe << rc << qpe.dagger()

def HHL_2x2_cao13_qiskit_corrected(A:Matrix=A12, b:Vector=b1) -> QCircuit:
  '''
    Same as `HHL_2x2_cao13_qiskit` but corrected errors in QPE part I think...
  '''

  if CHECK:
    assert any([A is someA for someA in [A12, A13]]), 'A should be chosen from [A12, A13]'
    assert (npl.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

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
      << U1(q3, pi/2).control(q2) \
      << U1(q3, pi)  .control(q1) \
      << SWAP(q1, q2) \
      << H(q2) \
      << U1(q2, -pi/2).control(q1) \
      << H(q1) \
      << (SWAP(q1, q2) if A is A12 else X(q1))

  # Step 3: controled-Y rotate θj = 2*arcsin(C/λi) ≈ 2*C/λi; 2^(1-r)*pi = 2*C
  rc = QCircuit() \
     << U3(q0, pi/8,  0, 0).control(q1) \
     << U3(q0, pi/16, 0, 0).control(q2)

  return enc_b << qpe << rc << qpe.dagger()

def HHL_2x2_pan13(A:Matrix=A12, b:Vector=b1) -> QCircuit:
  ''' 
    Implementation of the toy HHL circuit solving a minimal 2x2 system using only 4 qubits
    given in essay "Experimental realization of quantum algorithm for solving linear systems of equations" by Pan et al.
      - https://arxiv.org/abs/1302.1946
  '''

  if CHECK:
    assert A is A12, 'A must be A12'
    assert (npl.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

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
     << RY(q0,   pi/2**r).control(q1) \
     << RY(q0, 2*pi/2**r).control(q2)

  return enc_b << qpe << rc << qpe.dagger()

def HHL_2x2_cai13(A:Matrix=A12, b:Vector=bp, init:bool=False) -> QCircuit:
  '''
    Highly optimized 2x2 HHL circuit in experiments of real optical quantum systems
    given in essay "Experimental quantum computing to solve systems of linear equations" by Cai et al.
      - https://arxiv.org/abs/1302.4310
  '''

  if CHECK:
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
     << H_theta(q0, pi/ 8).control(q1) \
     << H_theta(q0, pi/16).control(q2) \
     << H(q2) \
     << H(q1)

  return qc

def HHL_2x2_zaman23(A:Matrix=Ah, b:Vector=bp) -> QCircuit:
  '''
    Implementation from "A Step-by-Step HHL Algorithm Walkthrough to Enhance Understanding of Critical Quantum Computing Concepts" by Zaman et al.
      - https://arxiv.org/pdf/2108.09004.pdf
  '''

  if CHECK:
    assert np.allclose(A, Ah)

  # q0: rotate
  # q1~q2: QPE
  # q3: ampl_enc |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  enc_b = encode_b(b, q3)

  pse = QCircuit() \
      << H(q1) \
      << H(q2) \
      << U4(pi, -0.5*pi, 0.5*pi, 0.75*pi, q3).control(q1) \
      << U4(pi, pi, 0, 0, q3).control(q2) \
      << H(q2) \
      << U1(q2, -0.5*pi).control(q1) \
      << H(q1) \
      << SWAP(q1, q2)

  cr = QCircuit() \
    << RY(q0, pi).control(q1) \
    << RY(q0, pi/3).control(q2)

  return enc_b << pse << cr << pse.dagger()

def HHL_2x2_qpanda(A:Matrix=A12, b:Vector=bp, ver:str='QPanda-dump-5') -> QCircuit:
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

  # q0: rotate
  # q1~q2: QPE
  # q3: ampl_enc |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  if ver == 'QRunes-code':    # this is impl of C++ code
    enc_b = QCircuit() \
          << RY(q3, pi/2)

    pse = QCircuit() \
        << H(q1) \
        << H(q2) \
        << RZ(q2, 0.75*pi) \
        << CU(pi, 1.5*pi, -pi/2, pi/2, q2, q3) \
        << RZ(q1, 1.5*pi) \
        << CU(pi, 1.5*pi,  pi,   pi/2, q1, q3) \
        << CNOT(q1, q2) \
        << CNOT(q2, q1) \
        << CNOT(q1, q2) \
        << H(q2) \
        << CU(-pi/4, -pi/2, 0, 0, q2, q1) \
        << H(q1)

    cr = QCircuit() \
      << X(q1) \
      << RY(q0, pi).control([q1, q2]) \
      << X(q1) \
      << X(q2) \
      << RY(q0, pi / 3).control([q1, q2]) \
      << X(q2) \
      << RY(q0, 0.679673818908).control([q1, q2])   # arcsin(1/3)

    cq = enc_b << pse << cr

  if ver == 'QRunes-graph':   # this is impl of the shown graph, just omits `enc_b` and differs in name with 'QRunes-code'
    pse = QCircuit() \
        << H(q1) \
        << H(q2) \
        << RZ(q2, 0.75*pi) \
        << RX(q3, -pi/2).control(q2) \
        << RZ(q1, 1.5*pi) \
        << RX(q3, -pi).control(q1) \
        << SWAP(q1, q2) \
        << H(q2) \
        << RZ(q1, -pi/2).control(q2) \
        << H(q1)

    cr = QCircuit() \
      << X(q1) \
      << RY(q0, pi).control([q1, q2]) \
      << X(q1) \
      << X(q2) \
      << RY(q0, pi / 3).control([q1, q2]) \
      << X(q2) \
      << RY(q0, 0.679673818908).control(q2)   # arcsin(1/3)

    cq = pse << cr << pse.dagger()

  if ver == 'QPanda-code':    # this is impl of C++ code
    enc_b = encode_b(b, q3)

    qpe = QCircuit() \
        << H(q1) \
        << H(q2) \
        << matrix_decompose(q3, spl.expm(1j*A*2*pi/1)).control(q2) \
        << matrix_decompose(q3, spl.expm(1j*A*2*pi/2)).control(q1) \
        << H(q2) \
        << CR(q1, q2, 2*pi/2) \
        << H(q1)

    cr = QCircuit() \
       << RY(q0, 2*np.arcsin(-1/3)).control([q1, q2]) \
       << RY(q0, 2*np.arcsin(-1/2)).control([q1, q2]) \
       << X(q1) << X(q2) \
       << RY(q0, 2*np.arcsin(-1/1)).control([q1, q2]) \
       << X(q1)

    cq = enc_b << qpe << cr << qpe.dagger()

  if ver == 'QPanda-dump-6':  # this is guessed impl from HHLAlg print(circuit) with 6-qubits
    # HHLAlg auto estimates n_quibits for QFT
    q4, q5 = list(qvm.qAlloc_many(2))

    # q0: |b>
    # q1~4: QFT
    # q5: ancilla

    # HHLAlg uses RY to encode
    enc_b = QCircuit() << RY(q0, 2*np.arccos(b[0]))

    # FIXME: when QOracle be replaced with `matrix_decompose()`, precision drops dramatically
    t = 2*pi / (1<<4)
    qpe = QCircuit() \
        << H([q1, q2, q3, q4]) \
        << encode_A(A, q0, t, 0).control(q4) \
        << encode_A(A, q0, t, 1).control(q3) \
        << encode_A(A, q0, t, 2).control(q2) \
        << encode_A(A, q0, t, 3).control(q1) \
        << H(q1) \
        << CR(q1, q2, pi/2).dagger() \
        << CR(q1, q3, pi/4).dagger() \
        << CR(q1, q4, pi/8).dagger() \
        << H(q2) \
        << CR(q2, q3, pi/2).dagger() \
        << CR(q2, q4, pi/4).dagger() \
        << H(q3) \
        << CR(q3, q4, pi/2).dagger() \
        << H(q4)

    cr = QCircuit() \
       << X([q2, q3, q4]) \
       << RY(q5, pi).control([q1, q2, q3, q4]) \
       << X([q1, q2]) \
       << RY(q5, pi/3).control([q1, q2, q3, q4]) \
       << X([q1]) \
       << RY(q5, 2*np.arcsin(1/3)).control([q1, q2, q3, q4]) \
       << X([q1, q2, q3]) \
       << RY(q5, 2*np.arcsin(1/4)).control([q1, q2, q3, q4]) \
       << X([q1]) \
       << RY(q5, 2*np.arcsin(1/5)).control([q1, q2, q3, q4]) \
       << X([q1, q2]) \
       << RY(q5, 2*np.arcsin(1/6)).control([q1, q2, q3, q4]) \
       << X([q1]) \
       << RY(q5, 2*np.arcsin(1/7)).control([q1, q2, q3, q4]) \
       << X([q1, q2, q3, q4]) \
       << RY(q5, -2*np.arcsin(1/8)).control([q1, q2, q3, q4]) \
       << X([q1]) \
       << RY(q5, -2*np.arcsin(1/7)).control([q1, q2, q3, q4]) \
       << X([q1, q2]) \
       << RY(q5, -2*np.arcsin(1/6)).control([q1, q2, q3, q4]) \
       << X([q1]) \
       << RY(q5, -2*np.arcsin(1/5)).control([q1, q2, q3, q4]) \
       << X([q1, q2, q3]) \
       << RY(q5, -2*np.arcsin(1/4)).control([q1, q2, q3, q4]) \
       << X([q1]) \
       << RY(q5, -2*np.arcsin(1/3)).control([q1, q2, q3, q4]) \
       << X([q1, q2]) \
       << RY(q5, -pi/3).control([q1, q2, q3, q4]) \
       << X([q1]) \
       << RY(q5, -pi).control([q1, q2, q3, q4])

    cq = enc_b << qpe << cr << qpe.dagger()

  if ver == 'QPanda-dump-5':  # this is guessed impl from HHLAlg print(circuit) with 5-qubits
    # HHLAlg auto estimates n_quibits for QFT, but seems at least 3 qubits
    q4 = qvm.qAlloc()

    # q0: |b>
    # q1~3: QFT (leave 1 for sign?? do not know why...)
    # q4: ancilla

    # HHLAlg uses RY to encode
    enc_b = QCircuit() << RY(q0, 2*np.arccos(b[0]))

    # FIXME: when QOracle be replaced with `matrix_decompose()`, precision drops dramatically
    t = 2*pi / (1<<3)     # theta = 2*pi / (1<<n_qft)
    qpe = QCircuit() \
        << H([q1, q2, q3]) \
        << encode_A(A, q0, t, 0).control(q3) \
        << encode_A(A, q0, t, 1).control(q2) \
        << encode_A(A, q0, t, 2).control(q1) \
        << H(q1) \
        << CR(q1, q2, pi/2).dagger() \
        << CR(q1, q3, pi/4).dagger() \
        << H(q2) \
        << CR(q2, q3, pi/2).dagger() \
        << H(q3)

    #cr = QCircuit() \
    #   << X([q2, q3]) \
    #   << RY(q4, pi).control([q1, q2, q3]) \
    #   << X([q1, q2]) \
    #   << RY(q4, pi/3).control([q1, q2, q3]) \
    #   << X([q1]) \
    #   << RY(q4, 2*np.arcsin(1/3)).control([q1, q2, q3]) \
    #   << X([q1, q2, q3]) \
    #   << RY(q4, -2*np.arcsin(1/4)).control([q1, q2, q3]) \
    #   << X([q1]) \
    #   << RY(q4, -2*np.arcsin(1/3)).control([q1, q2, q3]) \
    #   << X([q1, q2]) \
    #   << RY(q4, -pi/3).control([q1, q2, q3]) \
    #   << X([q1]) \
    #   << RY(q4, -pi).control([q1, q2, q3])

    cr = QCircuit() \
       << X([q2, q3]) \
       << RY(q4, pi).control([q1, q2, q3]) \
       << X([q1, q2]) \
       << RY(q4, pi/3).control([q1, q2, q3]) \
       << X([q3]) \
       << RY(q4, -pi/3).control([q1, q2, q3]) \
       << X([q1]) \
       << RY(q4, -pi).control([q1, q2, q3])

    cq = enc_b << qpe << cr << qpe.dagger()

  if ver == 'QPanda-dump-4':  # this is guessed impl for HHLAlg using 4-qubits (not really happen)
    # reordered to match the essays' convention
    # q0: |b>
    # q1~2: QFT
    # q3: ancilla

    # HHLAlg uses RY to encode
    enc_b = QCircuit() << RY(q0, 2*np.arccos(b[0]))

    t = 2*pi / (1<<2)
    qpe = QCircuit() \
        << H([q1, q2]) \
        << encode_A(A, q0, t, 0).control(q2) \
        << encode_A(A, q0, t, 1).control(q1) \
        << H(q1) \
        << CR(q1, q2, pi/2).dagger() \
        << H(q2)

    cr = QCircuit() \
       << X([q1]) \
       << RY(q3, pi).control([q1, q2]) \
       << X([q2]) \
       << RY(q3, pi/3).control([q1, q2]) \
       << X([q2]) \
       << RY(q3, -pi/3).control([q1, q2]) \
       << X([q1]) \
       << RY(q3, -pi).control([q1, q2])

    cq = enc_b << qpe << cr << qpe.dagger()

  return cq

def HHL_2x2_final(A:Matrix=A12, b:Vector=b1) -> QCircuit:
  '''
    My modified version, looking for best param according to Qiskit:
      - https://qiskit.org/textbook/ch-applications/hhl_tutorial.html
  '''

  if CHECK:
    assert np.allclose(A, A.conj().T), 'A should be a hermitian'
    assert (npl.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'

  # q0: perform RY to inverse λi
  # q1~q2: QPE scatter A => Σ|λi>, λi is integer approx to eigvals of A
  # q3: amplitude encode b => |b>
  q0, q1, q2, q3 = list(qvm.qAlloc_many(4))    # work around of tuple unpack

  # Step 1: prepare |b>, very toyish :(
  enc_b = encode_b(b, q3)

  # Step 2: QPE of A over |b>
  # following the approx formula: λj' = 2^n * (λj*t) / (2*pi)
  # in order to let λj' = λj, we have t = 2*pi / 2**n_qft = pi/2
  t = 2*pi / 2**2
  qpe = QCircuit() \
      << H([q1, q2]) \
      << encode_A(A, q3, t, 0).control(q2) \
      << encode_A(A, q3, t, 1).control(q1) \
      << H(q1) \
      << CR(q1, q2, pi/2).dagger() \
      << H(q2)

  if not 'debug QPE':
    qvm.directly_run(QProg() << enc_b << qpe)
    print(np.round(qvm.get_qstate(), 4).real)

  # Step 3: controled-Y rotate θj = 2*arcsin(C/λi) ≈ 2*C/λi; 2^(1-r)*pi = 2*C
  # NOTE: the rotation angles are accordingly esitimated by eigvals of A
  # note that RY(θ)|1> = [    // the more θ it rotates, the more away from |1> and shifts to |0>
  #   -sin(θ/2),
  #    cos(θ/2),
  # ]
  eigenvals, eigenvecs = npl.eig(A)
  lbd1, lbd2 = sorted(eigenvals)
  C = lbd1        # C is a normalizer being C <= min(λi)
  # we starts ancilla qubit from |1>, so add a sign to rotation angle θj
  cr = QCircuit() \
      << X(q0) \
      << RY(q0, -2*np.arcsin(C/lbd1)).control(q1) \
      << RY(q0, -2*np.arcsin(C/lbd2)).control(q2)

  if not 'debug rotate':
    qvm.directly_run(QProg() << enc_b << qpe << cr)
    print(np.round(qvm.get_qstate(), 4).real)

  return enc_b << qpe << cr << qpe.dagger()


def HHL(A:Matrix, b:Vector, HHL_cq:Callable, meth:str='hermitian', prt_ircode=False) -> Tuple[List[float], str]:
  '''
    Solving linear system equation `Ax = b` by quantum simulation in poly(logN) steps
      - https://arxiv.org/abs/0811.3171
      - https://arxiv.org/abs/2108.09004
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
  (Ar, b_n), stats = transform(A, b, meth=meth)

  if DEBUG and 'preprocess':
    print('Ar:') ; print(Ar)
    print('b_n:', b_n)
    if stats:
      D, b_norm = stats
      print('D:')  ; print(D)
      print('b_norm:', b_norm)

    x_hat = npl.solve(A,  b)      # answer of original equation
    y_hat = npl.solve(Ar, b_n)    # answer of transformed equation

  # Step 2: solving Ar * y = b_n in a quantum manner
  global qvm
  qvm = CPUQVM()
  qvm.init_qvm()

  hhl_cq = HHL_cq(Ar, b_n)

  if DEBUG and 'circuit':
    print(hhl_cq)

  prog = QProg() << hhl_cq
  qvm.directly_run(prog)
  y = project_q3(qvm.get_qstate(), is_first=HHL_cq is HHL_2x2_qpanda)

  ircode = ''
  if prt_ircode:
    try: ircode = to_originir(prog, qvm)
    except: pass
  #qvm.qFree_all(qvm.get_allocate_qubits())   # FIXME: buggy, stucks to SEGV
  qvm.finalize()

  # Step 3: inv-transform in a classical manner
  y = np.asarray(y, dtype=np.complex128)
  x = transform_inv(y, *stats, meth=meth)
  x = x.real.tolist()

  if DEBUG and 'postprocess':
    print('x_truth:', x_hat)
    print('y_truth:', y_hat)
    print('y_solut:', y)
    print('x_solut:', x)

  return x, ircode


def question1() -> Tuple[list, str]:
  return HHL(A, b, HHL_2x2_final, 'hermitian')


def go(A:Matrix, b:Vector, HHL_cq:Callable=HHL_2x2_cao13, meth:str='hermitian'):
  try:
    z = npl.solve(A, b)
  except npl.LinAlgError:   # ignore bad rand case if singular
    return

  x, _ = HHL(A, b, HHL_cq, meth=meth)
  print('  solut:', x)
  print('  truth:', z)

  if not 'show normed':
    print('  n_solut:', x / npl.norm(x))
    print('  n_truth:', z / npl.norm(z))


def benchmark(kind:str, eps=1e-2):
  circuits = [name for name in globals() if name.startswith('HHL_2x2_')]

  v_errors = [0.0] * len(circuits)    # error of value L2
  n_errors = [0.0] * len(circuits)    # error up to a normalization

  for _ in range(1000):
    if   kind == 'random':
      A_ = np.random.uniform(size=[2, 2], low=-1.0, high=1.0)
      b_ = np.random.uniform(size=[2],    low=-1.0, high=1.0)
    elif kind == 'target':     # around target case
      A_ = A + np.random.uniform(size=[2, 2], low=-1.0, high=1.0) * eps
      b_ = b + np.random.uniform(size=[2],    low=-1.0, high=1.0) * eps
    else: raise ValueError

    try:
      z = npl.solve(A_, b_)
      z_n = z / npl.norm(z)
    except npl.LinAlgError:
      continue

    for i, name in enumerate(circuits):
      x, _ = HHL(A_, b_, globals()[name])
      v_errors[i] += npl.norm(np.abs(z - np.asarray(x)))
      x_n = x / npl.norm(x)
      n_errors[i] += npl.norm(np.abs(z_n - x_n))
  
  for i, name in enumerate(circuits):
    print(f'{name}: {v_errors[i]} / {n_errors[i]}') 


if __name__ == '__main__':
  if not 'sanity check encode_b()':
    qvm = CPUQVM()
    qvm.init_qvm()
    q = qvm.qAlloc()
    for _ in range(10):
      b = np.random.uniform(size=[2], low=-1.0, high=1.0)
      b /= npl.norm(b)
      encode_b(b, q)
    # qvm.qFree(q)    # FIXME: buggy, stucks to SEGV
    qvm.finalize()

  if not 'sanity check eigen_A()':
    for A in [A12, A13, A23, A]:
      eigen_A(A)

  DEBUG = False

  # solve the specified target linear equation
  print('[target question] different methods')
  for name in [name for name in globals() if name.startswith('HHL_2x2_')]:
    print(name)
    go(A, b, globals()[name], meth='hermitian')

  HHL_cq = HHL_2x2_final
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

  # benchmark different circuits
  print('[benchmark random] error:')
  benchmark(kind='random')
  print('[benchmark target] error:')
  benchmark(kind='target', eps=1e-2)
