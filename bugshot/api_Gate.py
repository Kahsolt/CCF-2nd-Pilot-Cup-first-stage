#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *

# (11) API
assert QGate
# (2) info
QGate.gate_type
QGate.gate_matrix
QGate.get_qubits
# (5) control
QGate.control
QGate.set_control
QGate.get_control_qubits
QGate.get_control_qubit_num
QGate.get_target_qubit_num
# (3) dagger
QGate.dagger
QGate.set_dagger
QGate.is_dagger


''' Test '''
qvm = CPUQVM()
qvm.init_qvm()

q = qvm.qAlloc_many(2)
qv = QVec()   # buffer for readout

g = H(q[0])
print(g.gate_type())
print(g.gate_type() == HADAMARD_GATE)
print(g.gate_matrix())
print(g.get_qubits(qv))

print(g.control(q[1]))
print(g.set_control(q[1]))
print(g.get_control_qubits(qv))
print(g.get_control_qubit_num())
print(g.get_target_qubit_num())

print(g.dagger())
print(g.set_dagger(False))
print(g.is_dagger())
print(g.set_dagger(True))
print(g.is_dagger())

prog = QProg() << g
print(qvm.prob_run_list(prog, q))
qvm.directly_run(prog)
print(qvm.get_qstate())
