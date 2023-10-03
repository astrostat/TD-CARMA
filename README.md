# TD-CARMA: Painless, Accurate, and Scalable Estimates of Gravitational Lens Time Delays with Flexible CARMA Processes

- "main_gpcarma.py" : use to fit multiple TD-CARMA(p,q,m) models in parallel (but each individual TD-CARMA model on single core). Recommended for p <=3, q <=2, m <= 4.
- "new_td_gp.py": use to fit single TD-CARMA(p,q,m) model, using multiple cores. Recommended for p >= 4. [run using "mpiexec -n X python3 new_td_gp.py" where X is number of cores. Requires installation of MPI, see installation for pyMultiNest (https://johannesbuchner.github.io/PyMultiNest/)].

Link to the paper: https://iopscience.iop.org/article/10.3847/1538-4357/acbea1
