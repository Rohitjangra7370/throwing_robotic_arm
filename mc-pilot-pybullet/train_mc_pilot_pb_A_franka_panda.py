"""
Train the PB-A baseline policy for the Franka Panda arm.
"""

from train_mc_pilot_pb_arm import main


if __name__ == "__main__":
    import sys

    sys.argv = [sys.argv[0], "--robot", "franka_panda", *sys.argv[1:]]
    main()
