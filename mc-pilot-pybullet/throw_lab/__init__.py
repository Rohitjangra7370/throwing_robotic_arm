"""throw_lab -- separate research sandbox for throw trajectory generation.

Nothing in the shipped mc-pilot-pybullet pipeline imports this package. It
imports the shipped code (ArmController, OptimizedReleaseSolver, the Eq. 35
drag model) read-only, so every comparison runs against the real arm model.

Run everything from the mc-pilot-pybullet/ directory:
    /usr/bin/python3 -m throw_lab.bench --help
    /usr/bin/python3 -m pytest throw_lab/tests -q
"""
