from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, execute, IBMQ, transpile
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.providers.aer import AerSimulator

from qiskit.circuit.library import EfficientSU2

from qiskit.quantum_info import Pauli, Operator


import numpy as np
from numpy.linalg import eigh
import csv
import stim
from circuit_manipulation import *

from timeit import default_timer as timer


def get_ref_energy(coeffs, paulis, return_groundstate=False):
    """
    Compute theoretical minimum energy.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    return_groundstate (Bool): Whether to return groundstate.
    
    Returns:
    (Float) minimum energy (optionally also groundstate as array).
    """
    # the final operation
    final_op = None

    for ii, el in enumerate(paulis):
        if ii == 0:
            final_op = coeffs[ii]*Operator(Pauli(el))
        else:
            final_op += coeffs[ii]*Operator(Pauli(el))
   
    # compute the eigenvalues
    evals, evecs = eigh(final_op.data)
   
    # get the minimum eigenvalue
    min_eigenval = np.min(evals)
    if return_groundstate:
        return min_eigenval, evecs[:,0]
    else:
        return min_eigenval

def hartreefock(circuit, HF_bitstring=None, **kwargs):
    """
    Append the EfficientSU2 (full entanglement) ansatz to input circuit, inplace.
    circuit (QuantumCircuit).
    HF_bitstring (String): Bitstring to initialize to, e.g. "01101" -> |01101> (in Qiskit ordering |10110>)
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    """
    if HF_bitstring is None:
        return
    for i in range(len(HF_bitstring)):
        if HF_bitstring[i] == "1":
            circuit.x(i)

def efficientsu2_full(n_qubits, repetitions):
    """
    EfficientSU2 ansatz with full entanglement.
    n_qubits (Int): Number of qubits in circuit.
    repetitions (Int): # ansatz repetitions.

    Returns: 
    (QuantumCircuit, Int) (ansatz, #parameters).
    """
    ansatz = EfficientSU2(num_qubits=n_qubits, entanglement='full', reps=repetitions, insert_barriers=True)
    num_params_ansatz = len(ansatz.parameters)
    ansatz = ansatz.decompose()
    return ansatz, num_params_ansatz

def add_ansatz(circuit, ansatz_func, parameters, ansatz_reps=1, **kwargs):
    """
    Append an ansatz (full entanglement) to input circuit, inplace.
    circuit (QuantumCircuit).
    ansatz_func (Function): Defines the ansatz circuit. Returns (ansatz, #parameters).
    parameters (Iterable[Float]): VQE parameters.
    ansatz_reps (Int): # ansatz repetitions.
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    """
    n_qubits = circuit.num_qubits
    ansatz, _ = ansatz_func(n_qubits, ansatz_reps)
    ansatz.assign_parameters(parameters=parameters, inplace=True)
    circuit.compose(ansatz, inplace=True)

def vqe_circuit(n_qubits, parameters, hamiltonian, init_func=hartreefock, ansatz_func=efficientsu2_full, ansatz_reps=1, init_last=False, **kwargs):
    """
    Construct a single VQE circuit.
    n_qubits (Int): Number of qubits in circuit.
    parameters (Iterable[Float]): VQE parameters.
    hamiltonian (String): Pauli string to evaluate.
    initialization (Function): Takes QuantumCircuit and applies state initialization inplace.
    parametrization (Function): Takes QuantumCircuit and applies ansatz inplace.
    init_last (Bool): Whether initialization should come after (True) or before (False) ansatz.
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    
    Returns:
    (QuantumCircuit) VQE circuit for a specific Pauli string.
    """
    qr = QuantumRegister(n_qubits)
    cr = ClassicalRegister(n_qubits)
    circuit = QuantumCircuit(qr, cr)
    
    if not init_last:
        init_func(circuit, **kwargs)
    #append the circuit with the state preparation ansatz
    if parameters is not None:
        add_ansatz(circuit, ansatz_func, parameters, ansatz_reps, **kwargs)
    if init_last:
        init_func(circuit, **kwargs)
    
    #add the measurement operations
    for i, el in enumerate(hamiltonian):
        if el == 'I':
            #no measurement for identity
            continue
        elif el == 'Z':
            circuit.measure(qr[i], cr[i])
        elif el == 'X':
            circuit.u(np.pi/2, 0, np.pi, qr[i])
            circuit.measure(qr[i], cr[i])
        elif el == 'Y':
            circuit.u(np.pi/2, 0, np.pi/2, qr[i])
            circuit.measure(qr[i], cr[i])
    return circuit

def all_transpiled_vqe_circuits(n_qubits, parameters, paulis, backend, seed_transpiler=25, remove_barriers=True, **kwargs) -> dict:
    """
    Transpiles all VQE circuits for a specific backend efficienlty (uses the fact that structure is the same / same ansatz -> similar transpiled circuits)
    n_qubits (Int): Number of qubits in circuit.
    parameters (Iterable[Float]): VQE parameters.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    backend (IBM backend): Can be simulator, fake backend or real backend; only with mode = "device_execution".
    seed_transpiler (Int): Random seed for the transpiler. Default is 25 because favorite number of Jason D. Chadwick.
    remove_barriers (Bool): Whether to remove barriers.
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    
    Returns:
    List[QuantumCircuit] of all transpiled VQE circuits.
    """
    backend_qubits = backend.configuration().n_qubits
    circuit = vqe_circuit(n_qubits, parameters, n_qubits*'Z', **kwargs)
    if remove_barriers:
        circuit = RemoveBarriers()(circuit)
    # transpile one circuit
    t_circuit = transpile(circuit, backend, optimization_level=3, seed_transpiler=seed_transpiler)

    # get the mapping from virtual to physical
    virtual_to_physical_mapping = {}
    for inst in t_circuit:
        if inst[0].name == 'measure':
            virtual_to_physical_mapping[inst[2][0].index] = inst[1][0].index
    # remove final measurements
    t_circuit.remove_final_measurements()
    # create all transpiled circuits
    all_transpiled_circuits = []
    for pauli in paulis:
        new_circ = QuantumCircuit(backend_qubits, n_qubits)
        new_circ.compose(t_circuit, inplace = True)
        for idx, el in enumerate(pauli):
            if el == 'I':
                continue
            elif el == 'Z':
                new_circ.measure(virtual_to_physical_mapping[idx], idx)
            elif el == 'X':
                new_circ.rz(np.pi/2, virtual_to_physical_mapping[idx])
                new_circ.sx(virtual_to_physical_mapping[idx])
                new_circ.rz(np.pi/2, virtual_to_physical_mapping[idx])
                new_circ.measure(virtual_to_physical_mapping[idx], idx)
            elif el == 'Y':
                new_circ.sx(virtual_to_physical_mapping[idx])
                new_circ.rz(np.pi/2, virtual_to_physical_mapping[idx])
                new_circ.measure(virtual_to_physical_mapping[idx], idx)
        all_transpiled_circuits.append(new_circ)
    # print(all_transpiled_circuits[-2].draw(fold=-1))
    return all_transpiled_circuits

def compute_expectations(n_qubits, parameters, paulis, shots, backend, mode, **kwargs):
    """
    Compute the expection values of the Pauli strings.
    n_qubits (Int): Number of qubits in circuit.
    parameters (Iterable[Float]): VQE parameters.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    backend (IBM backend): Can be simulator, fake backend or real backend; only with mode = "device_execution".
    mode (String): ["no_noisy_sim", "device_execution", "noisy_sim"].
    shots (Int): Number of VQE circuit execution shots.
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    
    Returns:
    List[Float] of expection value for each Pauli string.
    """
    #evaluate the circuits
    if mode == 'no_noisy_sim':
        #get all the vqe circuits
        circuits = [vqe_circuit(n_qubits, parameters, pauli, **kwargs) for pauli in paulis]
        result = execute(circuits, backend=Aer.get_backend("qasm_simulator"), shots=shots).result()
    elif mode == 'device_execution':
        tcircs = all_transpiled_vqe_circuits(n_qubits, parameters, paulis, backend, **kwargs)
        job = execute(tcircs, backend=backend, shots=shots)
        result = job.result()
    elif mode == 'noisy_sim':
        sim_device = AerSimulator.from_backend(backend)
        tcircs = all_transpiled_vqe_circuits(n_qubits, parameters, paulis, backend, **kwargs)
        result = sim_device.run(tcircs, shots=shots).result()
    else:
        raise Exception('Invalid circuit execution mode')
    print("all circs run!")

    all_counts = []
    for __, _id in enumerate(paulis):
        if _id == len(_id)*'I':
            all_counts.append({len(_id)*'0':shots})
        else:
            all_counts.append(result.get_counts(__))
    
    #compute the expectations
    expectations = []
    for i, count in enumerate(all_counts):
        #initiate the expectation value to 0
        expectation_val = 0
        #compute the expectation
        for el in count.keys():
            sign = 1
            #change sign if there are an odd number of ones
            if el.count('1')%2 == 1:
                sign = -1
            expectation_val += sign*count[el]/shots
        expectations.append(expectation_val)
    return expectations

def vqe(n_qubits, parameters, coeffs, loss_filename=None, params_filename=None, **kwargs):
    """
    Compute the VQE loss/energy.
    n_qubits (Int): Number of qubits in circuit.
    parameters (Iterable[Float]): VQE parameters.ä
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    loss_filename (String): Path to save file for VQE loss/energy.
    params_filename (String): Path to save file for VQE parameters.
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    
    Returns:
    (Float) VQE energy. 
    """
    start = timer()
    expectations = compute_expectations(n_qubits, parameters, **kwargs)
    loss = np.inner(coeffs, expectations)
    end = timer()
    print(f'Loss computed by VQE is {loss}, in {end - start} s.')
    
    if loss_filename is not None:
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss])
    
    if params_filename is not None and parameters is not None:
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(parameters)
    return loss

def vqe_cafqa_stim(inputs, n_qubits, coeffs, paulis, init_func=hartreefock, ansatz_func=efficientsu2_full, ansatz_reps=1, init_last=False, loss_filename=None, params_filename=None, **kwargs):
    """
    Compute the CAFQA VQE loss/energy using stim.
    inputs (Dict): CAFQA VQE parameters (values in 0...3) as passed by hypermapper, e.g.: {"x0": 1, "x1": 0, "x2": 0, "x3": 2}
    n_qubits (Int): Number of qubits in circuit.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    initialization (Function): Takes QuantumCircuit and applies state initialization inplace.
    parametrization (Function): Takes QuantumCircuit and applies ansatz inplace.
    init_last (Bool): Whether initialization should come after (True) or before (False) ansatz.
    loss_filename (String): Path to save file for VQE loss/energy.
    params_filename (String): Path to save file for VQE parameters.
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.
    
    Returns:
    (Float) CAFQA VQE energy. 
    """
    start = timer()
    parameters = []
    # take the hypermapper parameters and convert them to vqe parameters
    for key in inputs:
        parameters.append(inputs[key]*(np.pi/2))

    vqe_qc = QuantumCircuit(n_qubits)
    if not init_last:
        init_func(vqe_qc, **kwargs)
    add_ansatz(vqe_qc, ansatz_func, parameters, ansatz_reps, **kwargs)
    if init_last:
        init_func(vqe_qc, **kwargs)
    vqe_qc_trans = transform_to_allowed_gates(vqe_qc)
    stim_qc = qiskit_to_stim(vqe_qc_trans)
    sim = stim.TableauSimulator()
    sim.do_circuit(stim_qc)
    pauli_expect = [sim.peek_observable_expectation(stim.PauliString(p)) for p in paulis]
    loss = np.dot(coeffs, pauli_expect)
    end = timer()
    print(f'Loss computed by CAFQA VQE is {loss}, in {end - start} s.')
    
    if loss_filename is not None:
        with open(loss_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([loss])
    
    if params_filename is not None and parameters is not None:
        with open(params_filename, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(parameters)
    return loss