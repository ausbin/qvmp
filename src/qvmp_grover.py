#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import math
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, assemble, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.algorithms import AmplificationProblem, Grover

# ## QROM

# In[2]:


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

# In[3]:


def inner_product(reg_sz, name="dot"):
    a = QuantumRegister(reg_sz, name=f"{name}_a")
    b = QuantumRegister(reg_sz, name=f"{name}_b")
    out = QuantumRegister(1, name=f"{name}_out")

    qc = QuantumCircuit(a, b, out)
    
    for i in range(reg_sz):
        qc.mct([a[i], b[i]], out)
        
    qc.name = name
    
    return qc

# ## Grover circuit

# In[4]:


def diffuser(nqubits):
    qc = QuantumCircuit(nqubits)

    for qubit in range(nqubits):
        qc.h(qubit)

    for qubit in range(nqubits):
        qc.x(qubit)

    qc.h(nqubits-1)
    qc.mct(list(range(nqubits-1)), nqubits-1)  # multi-controlled-toffoli
    qc.h(nqubits-1)

    for qubit in range(nqubits):
        qc.x(qubit)

    for qubit in range(nqubits):
        qc.h(qubit)

    U_s = qc.to_gate()
    U_s.name = "diffuser"
    return U_s

# In[5]:


def encode_vector(qc, qreg, vec):
    a = vec.reshape(-1)
    
    for i in range(len(a)):
        if a[i] == 1:
            qc.x(qreg[i])

# In[6]:


def gen_input(dim, num_incorrect):
    nrows, ncols = dim
    A = np.random.randint(0, 2, dim)
    y = np.random.randint(0, 2, (ncols,))
    z = A@y % 2

    idx = np.random.randint(0, nrows, (num_incorrect,))
    for i in idx:
        z[i] = not z[i]

    return A, y, z, idx

# In[7]:


def qvmp_grover_submatrix(A, y, z, num_solutions, num_iterations=None, with_amplitude_amplification=False):
    db = np.concatenate((A, z.reshape((1, -1)).T), axis=1)
    nrows, ncols = A.shape
    addr_size = math.ceil(math.log2(nrows))

    addr_qreg = QuantumRegister(addr_size, name="address")
    A_qreg = QuantumRegister(ncols, name="a")
    y_qreg = QuantumRegister(ncols, name="y")
    z_qreg = QuantumRegister(1, name="z")

    data_out = ClassicalRegister(addr_size, name="data")

    qc = QuantumCircuit(addr_qreg, A_qreg, y_qreg, z_qreg, data_out)

    qc.h(addr_qreg)
    encode_vector(qc, y_qreg, y)
    qc.barrier()

    qdb = qrom(db)
    dot = inner_product(ncols)

    if num_iterations is None:
        if num_solutions == 0:
            num_iterations = 1
        else:
            num_iterations = Grover.optimal_num_iterations(num_solutions, addr_size)
    
    num_aa_iterations = math.floor(math.sqrt(num_iterations))
    
    def do_grover_iterations():
        for i in range(num_iterations):
            qc.append(qdb, [*addr_qreg, *A_qreg, *z_qreg])
            qc.append(dot, [*A_qreg, *y_qreg, *z_qreg])

            qc.z(z_qreg)

            qc.append(dot.inverse(), [*A_qreg, *y_qreg, *z_qreg])
            qc.append(qdb.inverse(), [*addr_qreg, *A_qreg, *z_qreg])

            qc.barrier()

            qc.append(diffuser(addr_size), [*addr_qreg])
    
    if with_amplitude_amplification:
        for j in range(num_aa_iterations):
            qc.h(addr_qreg)
            qc.barrier()
            do_grover_iterations()
            qc.barrier()
            qc.append(diffuser(addr_size), [*addr_qreg])
    else:
        do_grover_iterations()
        
    qc.barrier()

    qc.measure([*addr_qreg], data_out)
    
    return qc, num_iterations

# In[10]:


# num_incorrect = 2
# dim = (4, 2)
# A, y, z, idx = gen_input(dim, num_incorrect)
# qc, num_iterations = qvmp_grover_submatrix(A, y, z, num_incorrect, num_iterations=8, with_amplitude_amplification=True)

# In[13]:


# qc.decompose().draw(fold=-1)

# In[46]:


# backend = AerSimulator(method="statevector", device="GPU")
# # backend = AerSimulator(method="matrix_product_state")
# qobj = assemble(transpile(qc, backend))
# result = backend.run(qobj).result()

# In[45]:


# plot_histogram(result.get_counts())
