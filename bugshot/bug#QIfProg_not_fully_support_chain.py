#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *

# BUG: QIfProg functionality is limited, not always correct when chaining up
# expected: support chainning QIf
# actually: get wrong answer in some cases

assert QIfProg

# ↓↓↓ Example ↓↓↓

qvm = CPUQVM()
qvm.init_qvm()

q = qvm.qAlloc_many(2)
c = qvm.cAlloc_many(2)
c[0].set_val(114)
c[1].set_val(514)

qif = QIfProg(c[0] > 233 or c[1] < 1919,      # 114 > 233 or 514 < 1919 => True
              QProg() << H(q[0]),             # <= sir, this way
              QProg() << X(q[0]))

qif2 = QIfProg(c[0] + 1 > c[1],               # 114 + 1 > 514 => False
               qif.get_false_branch(),
               QProg() << qif << CNOT(q[0], q[1]))     # <= this way

prog = QProg() << qif2
# FIXME: cannot print when condition is even const in compile-time
#print('prog_chain_if:')
#print(prog)
#print(qif.get_true_branch())
#print(qif.get_false_branch())
#print(qif2.get_true_branch())
#print(qif2.get_false_branch())

prog_eqv = QProg() << H(q[0]) << CNOT(q[0], q[1])
print('prog_eqv:')
print(prog_eqv)

# FIXME: should give out the same answer as equivalent one ↓↓↓
qvm.directly_run(prog)
s1 = qvm.get_qstate()
print(s1)
qvm.directly_run(prog_eqv)
s2 = qvm.get_qstate()
print(s2)

assert s1 == s2     # FIXME: AssertionError
