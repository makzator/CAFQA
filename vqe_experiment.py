import numpy as np

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter

from skquant.opt import minimize
import hypermapper
import json
import sys

from vqe_helpers import *
from circuit_manipulation import *


def molecule(atom_string, new_num_orbitals=None, **kwargs):
    """
    Compute Hamiltonian for molecule in qubit encoding using Qiskit Nature.
    atom_string (String): string to describe molecule, passed to PySCFDriver.
    new_num_orbitals (Int): Number of orbitals in active space (if None, use default result from PySCFDriver).
    kwargs (Dict): All the arguments that need to be passed on to the next function calls.

    Returns:
    (Iterable[Float], Iterable[String], String) (Pauli coefficients, Pauli strings, Hartree-Fock bitstring)
    """
    converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
    driver = PySCFDriver(
        atom=atom_string,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    problem = driver.run()
    if new_num_orbitals is not None:
        num_electrons = (problem.num_alpha, problem.num_beta)
        transformer = ActiveSpaceTransformer(num_electrons, new_num_orbitals)
        problem = transformer.transform(problem)
    ferOp = problem.hamiltonian.second_q_op()
    qubitOp = converter.convert(ferOp, problem.num_particles)
    initial_state = HartreeFock(
        problem.num_spatial_orbitals,
        problem.num_particles,
        converter
    )
    bitstring = "".join(["1" if bit else "0" for bit in initial_state._bitstr])
    # need to reverse order bc of qiskit endianness
    paulis = [x[::-1] for x in qubitOp.primitive.paulis.to_labels()]
    # add the shift as extra I pauli
    paulis.append("I"*len(paulis[0]))
    paulis = np.array(paulis)
    coeffs = list(qubitOp.primitive.coeffs)
    # add the shift (nuclear repulsion)
    coeffs.append(problem.nuclear_repulsion_energy)
    coeffs = np.array(coeffs).real
    return coeffs, paulis, bitstring

def run_vqe(n_qubits, coeffs, paulis, param_guess, budget, shots, mode, backend, save_dir, loss_file, params_file, vqe_kwargs):
    """
    Run VQE instance. Uses skquant for optimization.
    n_qubits (Int): Number of qubits in circuit.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    param_guess (Iterable[Float]): Initial guess for VQE parameters.
    budget (Int): Max number of optimization iterations.
    shots (Int): Number of VQE circuit execution shots.
    mode (String): ["no_noisy_sim", "device_execution", "noisy_sim"].
    backend (IBM backend): Can be simulator, fake backend or real backend; irrelevant with mode = "no_noisy_sim".
    save_dir (String): Save directory.
    loss_file (String): Name of save file for VQE loss/energy.
    params_file (String): Name of save file for VQE parameters.
    vqe_kwargs (Dict): Dictionary with additional keyword arguments for vqe() call.

    Returns:
    Tuple of energy estimate and optimized parameters.
    """
    # check right number of parameters given
    _, num_params = efficientsu2_full(n_qubits, vqe_kwargs["ansatz_reps"])
    if len(param_guess) == 0:
        param_guess = [0] * num_params
    assert len(param_guess) == num_params, f"Number of parameters given ({len(param_guess)}) does not match ansatz ({num_params})." 

    bounds = np.array([[0, np.pi*2]]*num_params)
    initial_point = np.array(param_guess)
    vqe_result = minimize(
            lambda c: vqe(
                n_qubits=n_qubits,
                parameters=c, 
                loss_filename=save_dir + "/" + loss_file,
                params_filename=save_dir + "/" + params_file,
                paulis=paulis, 
                coeffs=coeffs,
                shots=shots, 
                backend=backend, 
                mode=mode, 
                **vqe_kwargs
            ), 
            initial_point, 
            bounds, 
            budget, 
            method='imfil')
    energy_vqe = vqe_result[0].optval
    params_vqe = vqe_result[0].optpar
    return energy_vqe, params_vqe


def run_cafqa(n_qubits, coeffs, paulis, param_guess, budget, shots, mode, backend, save_dir, loss_file, params_file, vqe_kwargs):
    """
    Run CAFQA VQE instance. Uses stim for fast Clifford circuit simulation and hypermapper for discrete optimization.
    n_qubits (Int): Number of qubits in circuit.
    coeffs (Iterable[Float]): Pauli coefficients in Hamiltonian.
    paulis (Iterable[String]): Corresponding Pauli strings in Hamiltonian (same order as coeffs).
    param_guess (Iterable[0...3]): Initial guess for CAFQA VQE parameters, which are factors for pi/2. E.g. param_guess = [1,0,0,2,3,1] for 6-parameter VQE with real parameters [pi/2,0,0,pi,3pi/2,pi/2].
    budget (Int): Max number of optimization iterations.
    shots (Int): --Not relevant here--.
    mode (String): --Not relevant here--.
    backend (IBM backend): --Not relevant here--.
    save_dir (String): Save directory.
    loss_file (String): Name of save file for VQE loss/energy.
    params_file (String): Name of save file for VQE parameters.
    vqe_kwargs (Dict): Dictionary with additional keyword arguments for vqe_cafqa_stim() call.

    Returns:
    Tuple of energy estimate and optimized CAFQA parameters.
    """
    # check right number of parameters given
    _, num_params = efficientsu2_full(n_qubits, vqe_kwargs["ansatz_reps"])
    if len(param_guess) == 0:
        param_guess = [0] * num_params
    assert len(param_guess) == num_params, f"Number of parameters given ({len(param_guess)}) does not match ansatz ({num_params})." 

    hypermapper_config_path = save_dir + "/hypermapper_config.json"
    config = {}
    config["application_name"] = "cafqa_optimization"
    config["optimization_objectives"] = ["value"]
    number_of_RS = budget//1
    config["design_of_experiment"] = {}
    config["design_of_experiment"]["number_of_samples"] = number_of_RS
    config["optimization_iterations"] = budget
    config["models"] = {}
    config["models"]["model"] = "random_forest"
    config["input_parameters"] = {}
    config["print_best"] = True
    config["print_posterior_best"] = True
    for i in range(num_params):
        x = {}
        x["parameter_type"] = "ordinal"
        x["values"] = [0, 1, 2, 3]
        x["parameter_default"] = param_guess[i]
        config["input_parameters"]["x" + str(i)] = x
    config["log_file"] = save_dir + '/hypermapper_log.log'
    config["output_data_file"] = save_dir + "/hypermapper_output.csv"
    with open(hypermapper_config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    stdout = sys.stdout
    hypermapper.optimizer.optimize(
        hypermapper_config_path, 
        lambda x: vqe_cafqa_stim(
            inputs=x,
            n_qubits=n_qubits,
            loss_filename=save_dir + "/" + loss_file,
            params_filename=save_dir + "/" + params_file,
            paulis=paulis, 
            coeffs=coeffs,
            shots=shots, 
            backend=backend, 
            mode=mode, 
            **vqe_kwargs
        ))
    sys.stdout = stdout

    energy_cafqa = np.inf
    x_cafqa = None
    with open(config["log_file"]) as f:
        lines = f.readlines()
        counter = 0
        for idx, line in enumerate(lines[::-1]):
            if line[:16] == "Best point found" or line[:29] == "Minimum of the posterior mean":
                counter += 1
                parts = lines[-1-idx+2].split(",")
                energy = float(parts[-1])
                if energy < energy_cafqa:
                    energy_cafqa = energy
                    x_cafqa = [int(y) for y in parts[:-1]]
            if counter == 2:
                break
    return energy_cafqa, x_cafqa