"""The package works with numba absent, and the backend env var is honoured."""

import os
import subprocess
import sys

SNIPPET = (
    "import sys; sys.modules['numba'] = None\n"
    "import psnw_py\n"
    "assert not psnw_py.HAVE_NUMBA and psnw_py.DEFAULT_BACKEND == 'python'\n"
    "a = psnw_py.align('ACGTACGTTTACGTACGT', 'ACGAACGTACGTACGT')\n"
    "print(a.score, a.cigar)\n"
)


def _run(code, **env):
    e = dict(os.environ)
    e.update(env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=e)


def test_without_numba():
    p = _run(SNIPPET)
    assert p.returncode == 0, p.stderr
    assert p.stdout.split() == ["32", "7M2D9M"]


def test_backend_env_var():
    p = _run("import psnw_py; print(psnw_py.DEFAULT_BACKEND)", PSNW_PY_BACKEND="python")
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "python"
    p = _run("import psnw_py", PSNW_PY_BACKEND="fortran")
    assert p.returncode != 0
    p = _run(
        "class Boom:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'numba' or (name and name.startswith('numba.')):\n"
        "            raise ImportError('no numba')\n"
        "        return None\n"
        "import sys\n"
        "sys.meta_path.insert(0, Boom())\n"
        "import psnw_py\n",
        PSNW_PY_BACKEND="numba",
    )
    assert p.returncode != 0
    assert "numba" in (p.stderr + p.stdout).lower()
