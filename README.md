# CAFQA

Implementation of an interface for the Variational Quantum Eigensolver and the CAFQA scheme (https://github.com/rgokulsm/CAFQA).

CAFQA is a special case of VQE where the ansatz is built of Clifford gates only and the optimization is therefore performed over a discrete set. This implementation uses [stim](https://github.com/quantumlib/Stim) for fast Clifford circuit simulation and [HyperMapper](https://github.com/luinardi/hypermapper) for Bayesian Optimization over the discrete search space.
