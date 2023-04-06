#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

from pyqpanda import *

# (12) API
assert QProg
# (2) add elem
QProg.insert
QProg.__lshift__
# (4) iter
QProg.begin
QProg.end
QProg.head
QProg.last
# (6) info
QProg.get_max_qubit_addr
QProg.get_qgate_num
QProg.get_used_qubits
QProg.get_used_cbits
QProg.is_empty
QProg.is_measure_last_pos   # this is broken


''' Test '''
qvm = CPUQVM()
qvm.init_qvm()
q = qvm.qAlloc_many(2)
c = qvm.cAlloc_many(2)
qv = QVec()
cv = []

p = QProg()
assert p.is_empty()
#assert not p.is_measure_last_pos()
p << H(q[0])
p << X(q[1])
assert not p.is_empty()
#assert not p.is_measure_last_pos()
p << measure_all(q, c)
assert p.is_measure_last_pos()
print(p)

print(p.get_max_qubit_addr())
print(p.get_qgate_num())
print(p.get_used_qubits(qv))
print(len(p.get_used_qubits([*q, *q])))   # wierd behaviour
print(p.get_used_cbits(cv))
#assert cv
#assert qv[0] == q[0] and qv[1] == q[1]
#assert cv[0] == c[0] and cv[1] == c[1]

it = p.begin()
assert it
while it != p.end():
  print('>')
  it = it.get_next()
while it != p.begin():
  print('<')
  it = it.get_pre()

it = p.head()
assert it
while it != p.last():
  print('>')
  it = it.get_next()
while it != p.head():
  print('<')
  it = it.get_pre()
