import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_sessionstart(session):
    """Fail once, with the fix, instead of ~80 'Cannot load URDF file' errors.

    The Gen3 model must live inside the installed pybullet_data package; the stock
    pybullet wheel does not ship it. scripts/install_robot_assets.py puts it there.
    """
    import pybullet_data

    urdf = os.path.join(pybullet_data.getDataPath(), "kinova_gen3", "gen3.urdf")
    if not os.path.isfile(urdf):
        import pytest
        pytest.exit(
            f"Kinova Gen3 model not installed ({urdf} is missing).\n"
            "Run once per Python environment:  python3 scripts/install_robot_assets.py",
            returncode=3,
        )
