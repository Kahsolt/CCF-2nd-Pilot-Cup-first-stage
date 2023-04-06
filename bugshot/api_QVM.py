#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/02 

from pyqpanda import *

# (28) API
assert QuantumMachine
# (11) VM meta info
QuantumMachine.init_qvm                 # initQVM
QuantumMachine.finalize
QuantumMachine.set_configure
QuantumMachine.get_status               # getStatus, FIXME: this API raise a `TypeError: Unable to convert function return value to a Python type!`
QuantumMachine.init_state
QuantumMachine.get_qstate
QuantumMachine.get_allocate_cbits
QuantumMachine.get_allocate_cmem_num    # getAllocateCMem
QuantumMachine.get_allocate_qubits
QuantumMachine.get_allocate_qubit_num   # getAllocateQubitNum
QuantumMachine.get_gate_time_map
# (10) qubit/cbit alloc/freee
QuantumMachine.allocate_qubit_through_phy_address
QuantumMachine.allocate_qubit_through_vir_address
QuantumMachine.qAlloc
QuantumMachine.qAlloc_many
QuantumMachine.qFree
QuantumMachine.qFree_all
QuantumMachine.cAlloc
QuantumMachine.cAlloc_many
QuantumMachine.cFree
QuantumMachine.cFree_all
# run QProg (7)
QuantumMachine.directly_run
QuantumMachine.run_with_configuration
QuantumMachine.async_run
if 'async_run':
  QuantumMachine.get_processed_qgate_num
  QuantumMachine.is_async_finished
  QuantumMachine.get_async_result
QuantumMachine.get_expectation

# (10) API
assert CPUQVM
# (9) measure
CPUQVM.get_prob_dict        # run 过一次之后可以反复 get
CPUQVM.get_prob_list
CPUQVM.get_prob_tuple_list
CPUQVM.prob_run_dict        # := run(prog) + get_prob_dict(qv)
CPUQVM.prob_run_list
CPUQVM.prob_run_tuple_list
CPUQVM.quick_measure        # directly_run+quick_measure => run_with_configuration?
CPUQVM.pmeasure             # := prob_run_tuple_list
CPUQVM.pmeasure_no_index    # := prob_run_list
# (1) config
CPUQVM.set_max_threads

# (9) API, same like `CPUQVM` except no `set_max_threads`
assert CPUSingleThreadQVM
# (9) measure
CPUSingleThreadQVM.get_prob_dict
CPUSingleThreadQVM.get_prob_list
CPUSingleThreadQVM.get_prob_tuple_list
CPUSingleThreadQVM.prob_run_dict
CPUSingleThreadQVM.prob_run_list
CPUSingleThreadQVM.prob_run_tuple_list
CPUSingleThreadQVM.quick_measure
CPUSingleThreadQVM.pmeasure
CPUSingleThreadQVM.pmeasure_no_index

# (9) API, same like `CPUSingleThreadQVM`
assert GPUQVM
# (9) measure
GPUQVM.get_prob_dict
GPUQVM.get_prob_list
GPUQVM.get_prob_tuple_list
GPUQVM.prob_run_dict
GPUQVM.prob_run_list
GPUQVM.prob_run_tuple_list
GPUQVM.quick_measure
GPUQVM.pmeasure
GPUQVM.pmeasure_no_index

# (7) API
assert NoiseQVM
# (6) noise
NoiseQVM.set_noise_model
NoiseQVM.set_mixed_unitary_error
NoiseQVM.set_rotation_error
NoiseQVM.set_reset_error
NoiseQVM.set_measure_error
NoiseQVM.set_readout_error
# (1) config
NoiseQVM.set_max_threads

# (20) API
assert MPSQVM
# (13) measure
MPSQVM.get_prob_dict
MPSQVM.get_prob_list
MPSQVM.get_prob_tuple_list
MPSQVM.prob_run_dict
MPSQVM.prob_run_list
MPSQVM.prob_run_tuple_list
MPSQVM.quick_measure
MPSQVM.pmeasure
MPSQVM.pmeasure_no_index
MPSQVM.pmeasure_dec_index
MPSQVM.pmeasure_dec_subset
MPSQVM.pmeasure_bin_index
MPSQVM.pmeasure_bin_subset
# (7) noise
MPSQVM.add_single_noise_model
MPSQVM.set_noise_model
MPSQVM.set_mixed_unitary_error
MPSQVM.set_rotation_error
MPSQVM.set_reset_error
MPSQVM.set_measure_error
MPSQVM.set_readout_error

# (6) API
assert PartialAmpQVM
# (6) measure
PartialAmpQVM.run
PartialAmpQVM.get_prob_dict
PartialAmpQVM.prob_run_dict       # := run + get_prob_dict
PartialAmpQVM.pmeasure_subset
PartialAmpQVM.pmeasure_dec_index
PartialAmpQVM.pmeasure_bin_index

# (7) API
assert SingleAmpQVM
# (5) measure
SingleAmpQVM.run
SingleAmpQVM.get_prob_dict
SingleAmpQVM.prob_run_dict       # := run + get_prob_dict
SingleAmpQVM.pmeasure_dec_index
SingleAmpQVM.pmeasure_bin_index
# (2) misc
SingleAmpQVM.get_sequence
SingleAmpQVM.get_quick_map_vertice

# (29) API
assert QCloud
# (6) config
QCloud.set_qcloud_api
QCloud.set_compute_url
QCloud.set_inquire_url
QCloud.set_batch_compute_url
QCloud.set_batch_inquire_url
QCloud.set_noise_model
QCloud.get_state_fidelity
QCloud.get_state_tomography_density
# (3) hamiltion
QCloud.get_expectation_exec
QCloud.get_expectation_query
QCloud.get_expectation_commit
# (8) Full Amp QVM
QCloud.full_amplitude_measure
QCloud.full_amplitude_measure_batch
QCloud.full_amplitude_measure_batch_query
QCloud.full_amplitude_measure_batch_commit
QCloud.full_amplitude_pmeasure
QCloud.full_amplitude_pmeasure_batch
QCloud.full_amplitude_pmeasure_batch_query
QCloud.full_amplitude_pmeasure_batch_commit
# (2) Partial Amp QVM
QCloud.partial_amplitude_pmeasure
QCloud.partial_amplitude_pmeasure_batch
# (2) Single Amp QVM
QCloud.single_amplitude_pmeasure
QCloud.single_amplitude_pmeasure_batch
# (2) Noise QVM
QCloud.noise_measure
QCloud.noise_measure_batch
# (4) RealChip QM
QCloud.real_chip_measure
QCloud.real_chip_measure_batch
QCloud.real_chip_measure_batch_query
QCloud.real_chip_measure_batch_commit


''' Test '''
qvm = CPUQVM()
qvm.init_qvm()

# 概率测量 PMeasure: 线路中不能加Measure，获得理论概率
q = qvm.qAlloc_many(2)
prog: QProg = QProg() << H(q) << X(q)

print(prog.is_measure_last_pos())
it = prog.begin()
while it != prog.end():
  print(it.get_node_type())
  it = it.get_next()

r = qvm.prob_run_dict(prog, q)      # run + get_prob_dict
print('prob_run_dict:', r)
r = qvm.prob_run_list(prog, q)
print('prob_run_list:', r)
r = qvm.prob_run_tuple_list(prog, q)
print('prob_run_tuple_list:', r)

# these APIs are redundant: 
#r = qvm.get_prob_dict(q)
#print('get_prob_dict:', r)
#r = qvm.get_prob_list(q)
#print('get_prob_list:', r)
#r = qvm.get_prob_tuple_list(q)
#print('get_prob_tuple_list:', r)
#r = qvm.pmeasure(q)
#print('pmeasure:', r)
#r = qvm.pmeasure_no_index(q)
#print('pmeasure_no_index:', r)

qvm.qFree_all(q)
print()

# 量子测量 Measure: 线路中要加Measure，获得测量结果
q = qvm.qAlloc_many(2)
c = qvm.cAlloc_many(2)    # [::-1]
c2 = qvm.cAlloc_many(2)
r = qvm.quick_measure(q, 1000)
print('quick_measure:', r)
breakpoint()

prog = QProg() << H(q) #<< measure_all(q[1:], c[1:])

for i in range(10):
  r = qvm.directly_run(prog)
  print('directly_run:', r)
  print('c:', [bit.get_val() for bit in c])
  r = qvm.quick_measure(q, 1000)
  print('quick_measure:', r)
r = qvm.run_with_configuration(prog, shot=1000)      # why the fuck you need to pass in a cbit here??
print('run_with_configuration:', r)
print('c:', [bit.get_val() for bit in c])

qvm.qFree_all(q)
qvm.cFree_all(c)
