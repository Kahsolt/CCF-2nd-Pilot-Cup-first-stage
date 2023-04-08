#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/06 

from P2 import *

if 'target':
  A = A12
  b = b1
else:
  A = np.random.uniform(size=[2, 2], low=-1.0, high=1.0).astype(np.complex128)
  b = np.random.uniform(size=[2],    low=-1.0, high=1.0)
  A = (A + A.conj().T) / 2
  b = b / np.linalg.norm(b)

qvm = CPUQVM()
qvm.init_qvm()
algo = HHLAlg(qvm)
cq = algo.get_hhl_circuit(A.flatten(), b, precision_cnt=0)
print(cq)

qvm.directly_run(cq)
print(np.round(qvm.get_qstate(), 5))

if not 'HHLAlg uses QOracle, cannot dump':
  ircode = to_originir(QProg() << cq, qvm)
  print('ircode:', ircode)

print(algo.query_uesed_qubit_num())
print(algo.get_qubit_for_b())
print(algo.get_qubit_for_QFT())
print(algo.get_ancillary_qubit())
print(algo.get_amplification_factor())

qvm.finalize()
