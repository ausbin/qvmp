#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.compiler import assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector


# In[2]:


def run_with_aer_sim(qc, memory=0):
    backend = AerSimulator(max_memory_mb=memory)
    qobj = assemble(transpile(qc, backend))
    result = backend.run(qobj).result()
    return result


# ## QROM

# In[3]:


def qrom(database, name="db"):
    # Check if database is binary
    assert ((database == 0) | (database == 1)).all()
    
    def encode_row(i):
        row = database[i]
        for j in range(len(row)):
            if row[j] == 1:
                qc.mct(addr_qreg, data_qreg[j])

    def bit_diff_pos(a, b):
        return (addrs[i] ^ addrs[i-1]).bit_length() - 1

    def grey_code(n):
        return n ^ (n >> 1)
    
    nrows, ncols = database.shape
    addr_size = math.ceil(math.log2(nrows))
    addrs = [grey_code(n) for n in range(nrows)]
    
    addr_qreg = QuantumRegister(addr_size, name=f"{name}-address")
    data_qreg = QuantumRegister(ncols, name=f"{name}-data")
    
    qc = QuantumCircuit(addr_qreg, data_qreg)
    
    qc.x(addr_qreg)
    encode_row(0)
    
    qc.barrier()
    
    for i in range(1, len(addrs)):
        qc.x(addr_qreg[bit_diff_pos(addrs[i], addrs[i-1])])
        encode_row(addrs[i])
        qc.barrier()
    
    qc.name = name
    
    return qc


# ## Inner Product (out of place)

# In[4]:


def inner_product(reg_sz, name="dot"):
    a = QuantumRegister(reg_sz, name=f"{name}_a")
    b = QuantumRegister(reg_sz, name=f"{name}_b")
    out = QuantumRegister(1, name=f"{name}_out")

    qc = QuantumCircuit(a, b, out)
    
    for i in range(reg_sz):
        qc.mct([a[i], b[i]], out)
        
    qc.name = name
    
    return qc


# ## Grover Search

# In[5]:


from qiskit.algorithms import AmplificationProblem, Grover


# In[8]:


def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)
    # Apply transformation |s> -> |00..0> (H-gates)
    for qubit in range(nqubits):
        qc.h(qubit)
    # Apply transformation |00..0> -> |11..1> (X-gates)
    for qubit in range(nqubits):
        qc.x(qubit)
    # Do multi-controlled-Z gate
    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)
    # Apply transformation |11..1> -> |00..0>
    for qubit in range(nqubits):
        qc.x(qubit)
    # Apply transformation |00..0> -> |s>
    for qubit in range(nqubits):
        qc.h(qubit)
    # We will return the diffuser as a gate
    U_s = qc.to_gate()
    U_s.name = "diffuser"
    return U_s


# In[9]:


# A = np.array([[1, 1], [0, 1]])
# y = np.array([0, 1])
# z = np.array([1, 0])

# A = np.array([
#     [0, 1, 0, 0],
#     [1, 1, 1, 0],
#     [1, 0, 1, 1],
#     [1, 1, 1, 0]
# ])
# y = np.array([
#     0,
#     1,
#     1,
#     0
# ])
# z = np.array([
#     1,
#     0,
#     1,
#     1
# ])

def gen_input(sz, num_incorrect):
    A = np.random.randint(0, 2, (sz, sz))
    y = np.random.randint(0, 2, (sz,))
    z = A@y % 2
    
    idx = np.random.randint(0, sz, (num_incorrect,))
    for i in idx:
        z[i] = not z[i]

    return A, y, z

A, y, z = gen_input(32, 2)

# In[10]:


num_solutions = (A@y % 2 != z).sum()
print("Number of solutions = ", num_solutions)


# In[11]:


db = np.concatenate((A, z.reshape((1, -1)).T), axis=1)
nrows, ncols = A.shape
addr_size = math.ceil(math.log2(nrows))

addr_qreg = QuantumRegister(addr_size, name="address")
A_qreg = QuantumRegister(ncols, name="a")
y_qreg = QuantumRegister(nrows, name="y")
z_qreg = QuantumRegister(1, name="z")

data_out = ClassicalRegister(addr_size, name="data")

qc = QuantumCircuit(addr_qreg, A_qreg, y_qreg, z_qreg, data_out)

def encode_vector(qc, qreg, vec):
    a = vec.reshape(-1)
    
    for i in range(len(a)):
        if a[i] == 1:
            qc.x(qreg[i])

qc.h(addr_qreg)
encode_vector(qc, y_qreg, y)
qc.barrier()

qdb = qrom(db)
dot = inner_product(ncols)

if num_solutions == 0:
    num_iterations = 1
else:
    num_iterations = Grover.optimal_num_iterations(num_solutions, addr_size)

print("Number of iterations =", num_iterations)
    
for i in range(num_iterations):
    qc.append(qdb, [*addr_qreg, *A_qreg, *z_qreg])
    qc.append(dot, [*A_qreg, *y_qreg, *z_qreg])
    
#     s1 = Statevector.from_instruction(qc).to_dict()

    qc.z(z_qreg)
    
    qc.append(dot.inverse(), [*A_qreg, *y_qreg, *z_qreg])
    qc.append(qdb.inverse(), [*addr_qreg, *A_qreg, *z_qreg])
    
#     s2 = Statevector.from_instruction(qc).to_dict()

    qc.barrier()
    
    qc.append(diffuser(addr_size), [*addr_qreg])
qc.barrier()

qc.measure([*addr_qreg], data_out)

# print(s1)
# print(s2)

qc.draw(fold=-1)


# In[13]:

print("Num qubits = ", qc.num_qubits)

backend = AerSimulator(
    # max_memory_mb=2**38
    device="GPU",
    method="matrix_product_state"
)

print(backend.available_devices())

transpiled_qobj = transpile(
    qc,
    backend=backend,
)
print("Circuit transpiled")
qobj = assemble(transpiled_qobj, backend=backend, shots=1024)
print("Circuit assembled")
result = backend.run(qobj).result()

# In[14]:


# plot_histogram(result.get_counts())
print(result.get_counts())


# In[203]:
