#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *

# (12) API, gate-like QProg
assert QCircuit
# (2) add elem
QCircuit.insert
QCircuit.__lshift__
# (4) iter
QCircuit.begin
QCircuit.end
QCircuit.head
QCircuit.last
# (5) gate-like
QCircuit.control
QCircuit.set_control
QCircuit.dagger
QCircuit.set_dagger
QCircuit.is_empty


''' Test '''
qvm = CPUQVM()
qvm.init_qvm()
q = qvm.qAlloc_many(3)

cq = QCircuit()
assert cq.is_empty()
cq << RY(q[0], 1.2) \
   << RZ(q[1], -1.8) \
   << CNOT(q[1], q[0])
assert not cq.is_empty()

print('original:')
print(cq)

print('control:')
print(cq.control(q[2]))   # Control(q[2], cq), q2 over all cq as a gate
cq.set_control(q[2])
print('set_control:')
print(cq)

print('dagger:')
print(cq.dagger())
cq.set_dagger(False)
cq.set_dagger(True)
