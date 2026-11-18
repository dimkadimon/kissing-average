# Data

Two files are used by `code/lift.py --class` and are **not** redistributed here:

| file | what | where to get it |
|---|---|---|
| `external/p48p_gram.npy` | integral Gram matrix of the extremal even unimodular lattice P48p | Nebe and Sloane, *Catalogue of Lattices*, <https://math.rwth-aachen.de/~Gabriele.Nebe/LATTICES/> |
| `class_p48_7193.npy` | the 7,193-line class of minimal lines with |cGc'^T| <= 2 | Kamenetsky, <https://github.com/dimkadimon/KissingNumbers>, tag `v7` |

Neither is needed for `code/shell.py`, `code/graphs.py` or `code/records.py`; the class
measurement quoted in the README is the only result that depends on them, and it is
reproduced by placing the two files at the paths above and running

```sh
python code/lift.py --class
```
