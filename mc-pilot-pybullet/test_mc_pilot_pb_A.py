"""
Compatibility entrypoint for PB-A training across supported robot arms.

Examples:
  python test_mc_pilot_pb_A.py
  python test_mc_pilot_pb_A.py --robot franka_panda
  python test_mc_pilot_pb_A.py --robot xarm6 --seed 2 --num_trials 12

Legacy short flags from the original KUKA-only script are still accepted:
  python test_mc_pilot_pb_A.py -seed 1 -num_trials 10
"""

from train_mc_pilot_pb_arm import main


def _normalize_argv(argv):
    flag_map = {
        "-seed": "--seed",
        "-num_trials": "--num_trials",
    }
    normalized = [argv[0]]
    saw_robot = False
    for arg in argv[1:]:
        mapped = flag_map.get(arg, arg)
        if mapped == "--robot":
            saw_robot = True
        normalized.append(mapped)
    if not saw_robot:
        normalized[1:1] = ["--robot", "kuka_iiwa"]
    return normalized


if __name__ == "__main__":
    import sys

    sys.argv = _normalize_argv(sys.argv)
    main()
