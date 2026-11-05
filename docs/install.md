# Install

```bash
pip install psnw-py          # numpy only
pip install 'psnw-py[numba]' # recommended: compiled kernels
```

From a checkout:

```bash
make dev   # pip install -e '.[dev,docs]'
make test
```

Python 3.9 or newer. The environment variable `PSNW_PY_BACKEND=python|numba`
selects the default backend; `numba` fails if numba is not installed.
