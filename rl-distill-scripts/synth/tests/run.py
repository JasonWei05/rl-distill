"""The synth tests as a pack.py job (pack runs synth scripts): python tests/run.py [pytest args, test files relative to
this directory; default: all of it]. Installs pytest first where the environment lacks it (the ScaleTrain image).

    python tests/run.py -v -x -o faulthandler_timeout=600 test_fastrl.py test_model.py
"""

import os
import subprocess
import sys

if __name__ == "__main__":  # forkserver / spawn workers (data.build) re-import the main script
    try:
        import pytest
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
        import pytest
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = sys.argv[1:]
    sys.exit(pytest.main(args if any(a.endswith(".py") for a in args) else [*args, "."]))
