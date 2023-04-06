#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()
qubits = qvm.qAlloc_many(3)
cbits  = qvm.cAlloc_many(3)

qif = QIfProg(cbits[0] > cbits[1] and cbits[2] < cbits[1],
              QProg() << H(qubits[0]) << H(qubits[1]) << H(qubits[2]), 
              QProg() << H(qubits[0]) << CNOT(qubits[0], qubits[1]) << CNOT(qubits[1], qubits[2]))
prog = QProg() << qif

info = to_originir(prog, qvm)
print(info)

print()

prog = QProg() \
     << H(qubits) \
     << assign(cbits[0], cbits[0] + 1) \
     << Measure(qubits[1], cbits[1])
prog = QProg() << create_while_prog(cbits[0] < 10, prog)
info = to_originir(prog, qvm)
print(info)
