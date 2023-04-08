#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/03 

from typing import Tuple, List, Callable, Union

from pyqpanda import *
import numpy as np
np.set_printoptions(suppress=True, precision=7)
import numpy.linalg as npl
import scipy.linalg as spl

Matrix = np.ndarray
Vector = np.ndarray

# case of the special: `A12` is the original essay example, `b1` is the test case from Qiskit code
# NOTE: Aij is required to be hermitian (not needed to be unitary though)
r2 = np.sqrt(2)
pi = np.pi
A12 = np.asarray([      # eigval: 2, 1
  [3, 1],               # eigvec: [0.70710678, 0.70710678], [-0.70710678, 0.70710678]
  [1, 3],
]) / 2
b1 = np.asarray([0, 1]) # |b> = |1>

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


def project_q3(qstate:List[complex], is_first:bool=False) -> List[complex]:
  ''' gather all cases of ancilla qubit to be |1>, then pick out the target |q> only '''

  if is_first:
    # the first |q0> is the target qubit
    return qstate[len(qstate)//2:][:2]
  else:
    # the last |q3> is the target qubit
    return [qstate[0], qstate[len(qstate)//2]]


def transform(A:Matrix, b:Vector, meth:str='hermitian', A_r:Matrix=A12) -> Tuple[Tuple[Matrix, Vector], Tuple[Matrix, float]]:
  ''' Transforming an arbitray Ax = b to equivalent forms '''

  assert meth in ['reduce', 'hermitian']

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

  assert meth in ['reduce', 'hermitian']

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


def encode_b(b:Vector, q:Qubit) -> QCircuit:
  '''
    Amplitude encode a unit vector b = [b0, b1] to |b> = b0|0> + b1|1> use RY+Z gate, 
    to fix the lacking the coeff `e^(i*phi)` compared with U3 gate when b1 < 0
                         RY(theta) |0> = |b>
      [cos(theta/2), -sin(theta/2)][1] = [b0]
      [sin(theta/2),  cos(theta/2)][0] = [b1]
  '''
  if isinstance(b, list): b = np.asarray(b)
  assert isinstance(b, Vector), 'should be a Vector/np.ndarray'
  assert tuple(b.shape) == (2,), 'should be a vector lengthed 2'
  assert (npl.norm(b) - 1.0) < 1e-5, 'should be a unit vector'

  theta = 2 * np.arccos(b[0])
  cq = QCircuit() << RY(q, theta)
  if b[1] < 0: cq << Z(q)             # fix sign of |1> part

  return cq

def encode_A(A:Matrix, q:Qubit, theta:float=2*pi, k:float=0, encoder=QOracle) -> Union[QGate, QCircuit]:
  ''' Unitary generator encode hermitian A to a unitary up to power of 2: U^2^k = exp(iAθ)^2^k '''
  assert isinstance(A, Matrix), 'should be a Matrix/np.ndarray'
  assert tuple(A.shape) == (2, 2), 'should be a 2x2 matrix'
  assert np.allclose(A, A.conj().T), 'should be a hermitian matrix'
  assert encoder in [QOracle, matrix_decompose]

  u = spl.expm(1j* A * theta).astype(np.complex128)   # FIXME: explain why this is a valid unitary??
  assert np.allclose(u @ u.conj().T, np.eye(len(u)))

  return encoder(q, npl.matrix_power(u, 2**k).flatten())


def HHL_2x2_qpanda_5(A:Matrix=A12, b:Vector=b1, enc:str='oracle') -> QCircuit:
  ''' The guessed implementations from QPanda HHLAlg print(circuit) using 5-qubits '''

  # q0: |b>
  # q1~3: QFT (leave 1 for sign?? do not know why...)
  # q4: ancilla
  qv = qvm.qAlloc_many(5)
  q0, q1, q2, q3, q4 = [qv[i] for i in range(len(qv))]

  # HHLAlg uses RY to encode
  enc_b = QCircuit() << RY(q0, 2*np.arccos(b[0]))

  # FIXME: when QOracle be replaced with `matrix_decompose()`, precision drops dramatically
  t = 2*pi / (1<<3)     # theta = 2*pi / (1<<n_qft)
  encoder = QOracle if enc == 'oracle' else matrix_decompose
  qpe = QCircuit() \
      << H([q1, q2, q3]) \
      << encode_A(A, q0, t, 0, encoder).control(q3) \
      << encode_A(A, q0, t, 1, encoder).control(q2) \
      << encode_A(A, q0, t, 2, encoder).control(q1) \
      << H(q1) \
      << CR(q1, q2, pi/2).dagger() \
      << CR(q1, q3, pi/4).dagger() \
      << H(q2) \
      << CR(q2, q3, pi/2).dagger() \
      << H(q3)

  cr = QCircuit() \
      << X([q2, q3]) \
      << RY(q4, pi).control([q1, q2, q3]) \
      << X([q1, q2]) \
      << RY(q4, pi/3).control([q1, q2, q3]) \
      << X([q3]) \
      << RY(q4, -pi/3).control([q1, q2, q3]) \
      << X([q1]) \
      << RY(q4, -pi).control([q1, q2, q3])

  return enc_b << qpe << cr << qpe.dagger()

def HHL_2x2_ours(A:Matrix=A12, b:Vector=b1, enc:str='oracle', rot:str='eigval') -> QCircuit:
  '''
    My modified version, looking for best param according to Qiskit:
      - https://qiskit.org/textbook/ch-applications/hhl_tutorial.html
      - https://arxiv.org/pdf/2108.09004.pdf
  '''

  assert np.allclose(A, A.conj().T), 'A should be a hermitian'
  assert (npl.norm(b) - 1.0) < 1e-5, 'b should be a unit vector'
  assert enc in ['oracle', 'decompose', 'hard-coded']
  assert rot in ['eigval', 'approx']

  # q0: perform RY to inverse λi
  # q1~q2: QPE scatter A => Σ|λi>, λi is integer approx to eigvals of A
  # q3: amplitude encode b => |b>
  qv = qvm.qAlloc_many(4)
  q0, q1, q2, q3 = [qv[i] for i in range(len(qv))]

  # Step 1: prepare |b>, very toyish :(
  enc_b = encode_b(b, q3)

  # Step 2: QPE of A over |b>
  # following the approx formula: λj' = 2^n * (λj*t) / (2*pi)
  # in order to let λj' = λj, we have t = 2*pi / 2**n_qft = pi/2
  t = 2*pi / 2**2

  if enc == 'oracle':
    u2 = encode_A(A, q3, t, 0, QOracle)
    u1 = encode_A(A, q3, t, 1, QOracle)
  if enc == 'decompose':
    u2 = encode_A(A, q3, t, 0, matrix_decompose)
    u1 = encode_A(A, q3, t, 1, matrix_decompose)
  if enc == 'hard-coded':
    # The actual U(iAθ) for A=A12, θ=pi/2 is:
    #   [-1+i -1-i]
    #   [-1-i -1+i] / 2
    # the U(iAθ)^2^0 is the same as above, it can be decomposed as U4 and X 
    # and the U(iAθ)^2^1 is actually X gate, nice ;)
    u2 = QCircuit() << X(q3) << U4(-0.75*pi, -pi/2, pi/2, pi/2, q3)
    u1 = X(q3)

  qpe = QCircuit() \
      << H([q1, q2]) \
      << u2.control(q2) \
      << u1.control(q1) \
      << H(q1) \
      << CR(q1, q2, pi/2).dagger() \
      << H(q2)

  # Step 3: controled-Y rotate θj = 2*arcsin(C/λi) ≈ 2*C/λi; 2^(1-r)*pi = 2*C
  # note that RY(θ)|1> = [    // the more θ it rotates, the more away from |1> and shifts to |0>
  #   -sin(θ/2),
  #    cos(θ/2),
  # ]
  # we starts ancilla qubit from |1>, so add a sign to rotation angle θj

  if rot == 'eigval':
    # the rotation angles are accordingly esitimated by eigvals of A
    eigenvals, eigenvecs = npl.eig(A)
    lbd1, lbd2 = sorted(eigenvals)
    C = lbd1        # C is a normalizer being C <= min(λi)
    r1 = -2*np.arcsin(C/lbd1)
    r2 = -2*np.arcsin(C/lbd2)
  if rot == 'approx':
    # the rotation angles are generally approximated
    # for big endian |q1q2>, |00>=0.0, |10>=0.25, |01>=0.5, |11>=0.75
    r1 = -pi
    r2 = -pi/3

  cr = QCircuit() \
      << X(q0) \
      << RY(q0, r1).control(q1) \
      << RY(q0, r2).control(q2)

  return enc_b << qpe << cr << qpe.dagger()


def HHL(A:Matrix, b:Vector, HHL_cq:Callable, HHL_cq_args:tuple=tuple(), meth:str='hermitian', prt_ircode=False) -> Union[List[float], Tuple[List[float], str]]:
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

  # Step 2: solving Ar * y = b_n in a quantum manner
  global qvm
  qvm = CPUQVM()
  qvm.init_qvm()

  hhl_cq = HHL_cq(Ar, b_n, *HHL_cq_args)

  prog = QProg() << hhl_cq
  qvm.directly_run(prog)
  y = project_q3(qvm.get_qstate(), is_first=HHL_cq is HHL_2x2_qpanda_5)

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

  if prt_ircode:
    return x, ircode
  else:
    return x


def question1() -> Tuple[list, str]:
  return HHL(A, b, HHL_2x2_ours, ('hard-coded', 'approx'), meth='hermitian', prt_ircode=True)


def benchmark(kind:str, eps=1e-2):
  circuits = [
    'ours',
    'qpanda_5',
  ]

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
      x = HHL(A_, b_, globals()['HHL_2x2_' + name], meth='hermitian')
      v_errors[i] += npl.norm(np.abs(z - np.asarray(x)))
      x_n = x / npl.norm(x)
      n_errors[i] += npl.norm(np.abs(z_n - x_n))

  for i, name in enumerate(circuits):
    print(f'  {name}: {v_errors[i]} / {n_errors[i]}') 

def run_compares(A:Matrix, b:Vector, title='case'):
  print(f'{title}')
  print('  truth:         ', npl.solve(A, b))
  print('  ours (r+hc+a): ', HHL(A, b, HHL_2x2_ours, ('hard-coded', 'approx'), meth='reduce'))
  print('  ours (r+hc+e): ', HHL(A, b, HHL_2x2_ours, ('hard-coded', 'eigval'), meth='reduce'))
  print('  ours (r+d+a):  ', HHL(A, b, HHL_2x2_ours, ('decompose', 'approx'),  meth='reduce'))
  print('  ours (r+d+e):  ', HHL(A, b, HHL_2x2_ours, ('decompose', 'eigval'),  meth='reduce'))
  print('  ours (r+o+a):  ', HHL(A, b, HHL_2x2_ours, ('oracle', 'approx'),     meth='reduce'))
  print('  ours (r+o+e):  ', HHL(A, b, HHL_2x2_ours, ('oracle', 'eigval'),     meth='reduce'))
  print('  ours (h+hc+a): ', HHL(A, b, HHL_2x2_ours, ('hard-coded', 'approx'), meth='hermitian'))
  print('  ours (h+hc+e): ', HHL(A, b, HHL_2x2_ours, ('hard-coded', 'eigval'), meth='hermitian'))
  print('  ours (h+d+a):  ', HHL(A, b, HHL_2x2_ours, ('decompose', 'approx'),  meth='hermitian'))
  print('  ours (h+d+e):  ', HHL(A, b, HHL_2x2_ours, ('decompose', 'eigval'),  meth='hermitian'))
  print('  ours (h+o+a):  ', HHL(A, b, HHL_2x2_ours, ('oracle', 'approx'),     meth='hermitian'))
  print('  ours (h+o+e):  ', HHL(A, b, HHL_2x2_ours, ('oracle', 'eigval'),     meth='hermitian'))
  print('  qpanda_5 (h+d):', HHL(A, b, HHL_2x2_qpanda_5, ('decompose',), meth='hermitian'))
  print('  qpanda_5 (h+o):', HHL(A, b, HHL_2x2_qpanda_5, ('oracle',),    meth='hermitian'))
  print('  qpanda_alg:    ', np.asarray(HHL_solve_linear_equations((A.conj().T @ A).astype(np.complex128).flatten(), A.conj().T @ b)).real)
  print()


if __name__ == '__main__':
  # solve the specified target linear equation
  run_compares(A, b, '[target question]')

  # benchmark error of different circuits
  print('[benchmark random] L1 error / L1 error after norm:')
  benchmark(kind='random')
  print('[benchmark target] L1 error / L1 error after norm:')
  benchmark(kind='target', eps=1e-2)
  print()

  # solve random linear equations
  A = np.random.uniform(size=[2, 2], low=-1.0, high=1.0)
  b = np.random.uniform(size=[2],    low=-1.0, high=1.0)
  run_compares(A, b, '[random question]')
  print()

  # test the question API
  _, ircode = question1()
  print(ircode)
