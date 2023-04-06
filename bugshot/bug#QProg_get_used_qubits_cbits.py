#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *
import numpy as np

# BUG: QProg API `get_used_qubits()` and `get_used_cbits()` signature is unreasonable and behaves wierd
# location: include\Core\QuantumCircuit\QProgram.h:58-70
# expected: do not need to pass `QVec` or `List[ClassicalCondition]` as arguments
# actually: forcely accepting these unneccesary auguments

# FIXME: 
#  - used_qubits and used_cbits should be calculated **innerly**, no depending on any inputs, just remove the unneccesary parameters!!!
#  - get_used_cbits(cv) dose not read out the cv!!

# ↓↓↓ Example ↓↓↓

qvm = CPUQVM()
qvm.init_qvm()
q = qvm.qAlloc_many(2)
c = qvm.cAlloc_many(2)
qv = QVec()
cv = []

prog: QProg = QProg() \
     << X(q[0]) \
     << H(q[1]) \
     << measure_all(q, c)

print(len(prog.get_used_qubits(qv)))   # FIXME: should NOT accept `qv`
print(len(prog.get_used_cbits(cv)))    # FIXME: should NOT accept `cv`
assert len(qv) != 0                    # correctly, reads out to qv
assert len(cv) == 0                    # FIXME: wrong, did not read out to cv!!!
print(len(prog.get_used_qubits(q)))    # should NOT accept `q`
print(len(prog.get_used_cbits(c)))     # should NOT accept `c`

print()

# FIXME: these are insane calls, it gives out non-interpretible results
# should be prevented !! 
print(len(prog.get_used_qubits([])))
print(len(prog.get_used_cbits([])))
print(len(prog.get_used_qubits([*q, *q, *q])))
print(len(prog.get_used_cbits([*c, *c, *c])))
