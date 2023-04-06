#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *

# (3) API
assert QIfProg
QIfProg.get_classical_condition
QIfProg.get_true_branch
QIfProg.get_false_branch


''' Test '''
qvm = CPUQVM()
qvm.init_qvm()

q = qvm.qAlloc_many(2)
c = qvm.cAlloc_many(2)
c[0].set_val(114)
c[1].set_val(514)

qif = QIfProg(c[0] > 233 or c[1] < 1919,      # 114 > 233 or 514 < 1919 => True
              QProg() << H(q[0]),             # <= sir, this way
              QProg() << X(q[0]))

print('qif:')
print(QProg() << qif)
print('cond:')
print(qif.get_classical_condition())
print('branch true:')
print(qif.get_true_branch())
print('branch false:')
print(qif.get_false_branch())


qif2 = QIfProg(c[0] + 1 > c[1],               # 114 + 1 > 514 => False
               qif.get_false_branch(),
               QProg() << qif << CNOT(q[0], q[1]))     # <= this way

# cannot print if has undecided QIf
#print(QProg() << qif2)
#print(qif2.get_classical_condition())
#print(qif2.get_true_branch())
#print(qif2.get_false_branch())

prog = QProg() << qif2
# cannot print if has undecided QIf
#print('prog:')
#print(prog)
qvm.directly_run(prog)
print(qvm.get_qstate())

prog_eqv = QProg() << H(q[0]) << CNOT(q[0], q[1])
print('prog_eqv:')
print(prog_eqv)
qvm.directly_run(prog_eqv)
print(qvm.get_qstate())
