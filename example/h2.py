import numpy as np
from qiskit.providers.fake_provider import FakeMumbai
import sys
sys.path.append("../")

from vqe_experiment import *


def main():
    bond_length = 1.0
    # H2 molecule string
    atom_string = f"H 0 0 0; H 0 0 {bond_length}"
    num_orbitals = 2
    coeffs, paulis, HF_bitstring = molecule(atom_string, num_orbitals)
    n_qubits = len(paulis[0])

    save_dir = "./"
    result_file = "result.txt"
    budget = 500
    vqe_kwargs = {
        "ansatz_reps": 2,
        "init_last": False,
        "HF_bitstring": HF_bitstring
    }

    # run CAFQA
    cafqa_guess = [] # will start from all 0 parameters
    loss_file = "cafqa_loss.txt"
    params_file = "cafqa_params.txt"
    cafqa_energy, cafqa_params = run_cafqa(
        n_qubits=n_qubits,
        coeffs=coeffs,
        paulis=paulis,
        param_guess=cafqa_guess,
        budget=budget,
        shots=None,
        mode=None,
        backend=None,
        save_dir=save_dir,
        loss_file=loss_file,
        params_file=params_file,
        vqe_kwargs=vqe_kwargs
    )
    with open(save_dir + result_file, "w") as res_file:
        res_file.write(f"CAFQA energy:\n{cafqa_energy}\n")
        res_file.write(f"CAFQA params (x pi/2):\n{np.array(cafqa_params)}\n\n")

    
    # VQE with CAFQA initialization
    shots = 8192
    loss_file = "vqe_loss.txt"
    params_file = "vqe_params.txt"
    vqe_energy, vqe_params = run_vqe(
        n_qubits=n_qubits,
        coeffs=coeffs,
        paulis=paulis,
        param_guess=np.array(cafqa_params)*np.pi/2,
        budget=budget,
        shots=shots,
        mode="device_execution",
        backend=FakeMumbai(),
        save_dir=save_dir,
        loss_file=loss_file,
        params_file=params_file,
        vqe_kwargs=vqe_kwargs
    )
    with open(save_dir + result_file, "a") as res_file:
        res_file.write(f"VQE energy:\n{vqe_energy}\n")
        res_file.write(f"VQE params (x pi/2):\n{np.array(vqe_params)}\n\n")


if __name__ == "__main__":
    main()