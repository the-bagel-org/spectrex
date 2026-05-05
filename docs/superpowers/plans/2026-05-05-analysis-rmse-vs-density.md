# Analysis: RMSE vs Source Density Notebook

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `notebooks/analysis_rmse_vs_density.ipynb` — a pre-executed notebook that sweeps source density from 1 to 20 sources, runs 10 trials per density, and plots mean ± std RMSE vs source count.

**Architecture:** The notebook is self-contained. A local `run_pipeline()` helper generates mock data and runs `SciPySparseOperator` + `SpectralSolver` (Phase 1 tools). The operator is built once for a 50×20 image and cached as `notebooks/analysis_operator_cache.npz` (gitignored). The notebook must be pre-executed before committing (outputs committed; `myst-nb` configured `nb_execution_mode = "off"`).

**Tech Stack:** `spectrex` (Phase 1 API), `numpy`, `matplotlib`, `jupyter`, `nbformat`

---

## File Map

| Path | Action | Purpose |
|---|---|---|
| `notebooks/analysis_rmse_vs_density.ipynb` | Create | The analysis notebook |
| `notebooks/analysis_operator_cache.npz` | Generate (gitignored) | Cached operator for the analysis image |
| `.gitignore` | Modify | Add `notebooks/analysis_operator_cache.npz` |

---

### Task 0: Worktree + gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Create feature worktree**

```bash
git worktree add .worktrees/analysis-rmse -b feature/analysis-rmse
```

- [ ] **Step 2: Add cache file to `.gitignore`**

Open `.gitignore` and add the following line in the `notebooks/` section (near the existing `notebooks/operator_cache.npz` entry):

```
notebooks/analysis_operator_cache.npz
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore analysis notebook operator cache"
```

---

### Task 1: Create the notebook

**Files:**
- Create: `notebooks/analysis_rmse_vs_density.ipynb`

Create the notebook using Python (run this script from the repo root):

- [ ] **Step 1: Write the notebook creation script**

Save as `/tmp/create_analysis_nb.py` and run it:

```python
"""Create notebooks/analysis_rmse_vs_density.ipynb."""

import json
from pathlib import Path

cells = []


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells.append(md(
    "# RMSE vs Source Density\n\n"
    "Sweep the number of sources from 1 to 20 and measure extraction RMSE "
    "over 10 independent trials per density using the Phase 1 "
    "`SciPySparseOperator` + `SpectralSolver` pipeline."
))

cells.append(code(
    "from __future__ import annotations\n\n"
    "import warnings\n"
    "from pathlib import Path\n\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n\n"
    "import spectrex\n"
    "from spectrex import (\n"
    "    EigenspectraBasis,\n"
    "    InstrumentConfig,\n"
    "    NoiseModel,\n"
    "    SciPySparseOperator,\n"
    "    SpectralSolver,\n"
    ")\n\n"
    "warnings.filterwarnings('ignore')\n\n"
    "# Paths — notebook assumed to live in notebooks/ inside the repo root\n"
    "NOTEBOOK_DIR = Path.cwd()\n"
    "REPO = NOTEBOOK_DIR.parent\n"
    "TESTDATA = REPO / 'testdata'\n"
    "CACHE_PATH = NOTEBOOK_DIR / 'analysis_operator_cache.npz'\n"
    "COLD_START = not CACHE_PATH.exists()\n"
    f"print(f'spectrex version: {{spectrex.__version__}}')\n"
    "print(f'Cold start: {COLD_START}')"
))

cells.append(md("## 1. Build or Load Operator"))

cells.append(code(
    "config = InstrumentConfig.from_files(\n"
    "    conf_path=TESTDATA / 'Config Files' / 'GR150R.F150W.220725.conf',\n"
    "    wavelengthrange_path=TESTDATA / 'jwst_niriss_wavelengthrange_0002.asdf',\n"
    "    sensitivity_dir=TESTDATA / 'SenseConfig' / 'wfss-grism-configuration',\n"
    "    filter_name='F150W',\n"
    ")\n"
    "basis = EigenspectraBasis.from_csv(\n"
    "    TESTDATA / 'eigenspectra_kurucz.csv',\n"
    "    config.wavelengths,\n"
    ")\n\n"
    "IMAGE_SHAPE = (50, 20)\n\n"
    "if COLD_START:\n"
    "    print('Building operator (cold start)…')\n"
    "    op = SciPySparseOperator.build(config, basis, IMAGE_SHAPE)\n"
    "    op.save(CACHE_PATH)\n"
    "    print(f'Saved to {CACHE_PATH}')\n"
    "else:\n"
    "    print('Loading cached operator…')\n"
    "    op = SciPySparseOperator.load(CACHE_PATH)\n"
    "    print('Done.')\n\n"
    "n_pix = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]\n"
    "n_comp = basis.n_components\n"
    "print(f'Image shape: {IMAGE_SHAPE}')\n"
    "print(f'n_coefficients: {op.n_coefficients}, n_components: {n_comp}')"
))

cells.append(md("## 2. Pipeline Helper"))

cells.append(code(
    "def run_pipeline(\n"
    "    op: SciPySparseOperator,\n"
    "    basis: EigenspectraBasis,\n"
    "    image_shape: tuple[int, int],\n"
    "    source_pixels: list[int],\n"
    "    rng: np.random.Generator,\n"
    "    noise_model: NoiseModel,\n"
    "    regularisation: float = 1e-2,\n"
    ") -> dict:\n"
    '    """Run one mock extraction trial for the given source pixel positions.\n\n'
    "    Parameters\n"
    "    ----------\n"
    "    op : SciPySparseOperator\n"
    "        Pre-built forward operator for the full image.\n"
    "    basis : EigenspectraBasis\n"
    "        Eigenspectra basis (used for n_components only).\n"
    "    image_shape : tuple[int, int]\n"
    "        ``(n_rows, n_cols)``.\n"
    "    source_pixels : list[int]\n"
    "        Flat pixel indices (row * n_cols + col) for each active source.\n"
    "    rng : np.random.Generator\n"
    "        NumPy random generator.\n"
    "    noise_model : NoiseModel\n"
    "        Noise model for mock observations and solve weighting.\n"
    "    regularisation : float\n"
    "        Tikhonov regularisation λ. Default 1e-2.\n\n"
    "    Returns\n"
    "    -------\n"
    "    dict\n"
    "        Keys: ``rmse``, ``n_sources``, ``a_true``, ``a_rec``.\n"
    '    """\n'
    "    n_pix_ = image_shape[0] * image_shape[1]\n"
    "    n_comp_ = basis.n_components\n\n"
    "    # True coefficients: non-zero only at source pixel blocks\n"
    "    a_true = np.zeros(n_pix_ * n_comp_)\n"
    "    for p in source_pixels:\n"
    "        a_true[p * n_comp_ : (p + 1) * n_comp_] = rng.standard_normal(n_comp_)\n\n"
    "    # Forward model → noisy observation\n"
    "    f_clean = op.apply(a_true).reshape(image_shape)\n"
    "    f_noisy = noise_model.sample(f_clean, rng)\n\n"
    "    # Support mask: True at coefficient blocks of active sources\n"
    "    mask = np.zeros(n_pix_ * n_comp_, dtype=bool)\n"
    "    for p in source_pixels:\n"
    "        mask[p * n_comp_ : (p + 1) * n_comp_] = True\n\n"
    "    # Solve\n"
    "    solver = SpectralSolver(\n"
    "        op, noise_model=noise_model, regularisation=regularisation\n"
    "    )\n"
    "    a_rec = solver.solve(f_noisy, support_mask=mask)\n\n"
    "    # RMSE on active sources only\n"
    "    rmse = float(np.sqrt(np.mean((a_rec[mask] - a_true[mask]) ** 2)))\n"
    "    return {\n"
    "        'rmse': rmse,\n"
    "        'n_sources': len(source_pixels),\n"
    "        'a_true': a_true,\n"
    "        'a_rec': a_rec,\n"
    "    }"
))

cells.append(md("## 3. Density Sweep"))

cells.append(code(
    "N_SOURCES_GRID = [1, 2, 3, 5, 8, 10, 15, 20]\n"
    "N_TRIALS = 10\n"
    "REGULARISATION = 1e-2\n"
    "NOISE_MODEL = NoiseModel(read_noise=5.0)\n"
    "MASTER_RNG = np.random.default_rng(2026)"
))

cells.append(code(
    "results: dict[int, list[float]] = {n: [] for n in N_SOURCES_GRID}\n\n"
    "for n_src in N_SOURCES_GRID:\n"
    "    for trial in range(N_TRIALS):\n"
    "        rng = np.random.default_rng(MASTER_RNG.integers(0, 2**31))\n"
    "        source_pixels = rng.choice(n_pix, size=n_src, replace=False).tolist()\n"
    "        res = run_pipeline(\n"
    "            op, basis, IMAGE_SHAPE, source_pixels, rng,\n"
    "            NOISE_MODEL, REGULARISATION,\n"
    "        )\n"
    "        results[n_src].append(res['rmse'])\n"
    "    mean_ = np.mean(results[n_src])\n"
    "    std_ = np.std(results[n_src])\n"
    "    print(f'n_sources={n_src:3d}: mean RMSE = {mean_:.4f} ± {std_:.4f}')"
))

cells.append(md("## 4. Results"))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(7, 4))\n\n"
    "ns_arr = np.array(N_SOURCES_GRID)\n"
    "means = np.array([np.mean(results[n]) for n in N_SOURCES_GRID])\n"
    "stds  = np.array([np.std(results[n])  for n in N_SOURCES_GRID])\n\n"
    "ax.fill_between(ns_arr, means - stds, means + stds,\n"
    "                alpha=0.3, label='±1σ')\n"
    "ax.plot(ns_arr, means, 'o-', label='Mean RMSE')\n"
    "ax.set_xlabel('Number of sources')\n"
    "ax.set_ylabel('RMSE (coefficient units)')\n"
    "ax.set_title('Extraction RMSE vs Source Density')\n"
    "ax.legend()\n"
    "ax.grid(True, alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md(
    "## 5. Observations\n\n"
    "*(Fill in after executing the notebook.)*\n\n"
    "- Expected: RMSE rises with source density as trace overlap increases.\n"
    "- At low density (1–3 sources), the problem is well-conditioned and "
    "RMSE should be near the noise floor.\n"
    "- The regularisation parameter `λ = 1e-2` is held fixed; a separate "
    "λ-sweep is left as a follow-up."
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "cells": cells,
}

out = Path("notebooks/analysis_rmse_vs_density.ipynb")
out.write_text(json.dumps(nb, indent=1))
print(f"Written: {out}")
```

- [ ] **Step 2: Run the creation script from repo root**

```bash
python /tmp/create_analysis_nb.py
```

Expected: `Written: notebooks/analysis_rmse_vs_density.ipynb`

- [ ] **Step 3: Verify the file exists**

```bash
python -c "import json,pathlib; nb=json.loads(pathlib.Path('notebooks/analysis_rmse_vs_density.ipynb').read_text()); print(f'{len(nb[\"cells\"])} cells')"
```

Expected: `11 cells`

- [ ] **Step 4: Commit unexecuted notebook**

```bash
git add notebooks/analysis_rmse_vs_density.ipynb .gitignore
git commit -m "feat: analysis_rmse_vs_density notebook skeleton (unexecuted)"
```

---

### Task 2: Execute and commit

**Files:**
- Modify: `notebooks/analysis_rmse_vs_density.ipynb`

- [ ] **Step 1: Execute the notebook (from the `notebooks/` directory)**

```bash
cd notebooks && jupyter nbconvert --to notebook --execute \
    analysis_rmse_vs_density.ipynb \
    --output analysis_rmse_vs_density.ipynb \
    --ExecutePreprocessor.timeout=600
```

Expected: completes without errors; the `.ipynb` file now has cell outputs.

- [ ] **Step 2: Verify outputs are present**

```bash
python -c "
import json, pathlib
nb = json.loads(pathlib.Path('notebooks/analysis_rmse_vs_density.ipynb').read_text())
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
cells_with_output = [c for c in code_cells if c['outputs']]
print(f'{len(cells_with_output)} / {len(code_cells)} code cells have outputs')
"
```

Expected: at least 4 of 7 code cells have outputs (imports, build, sweep, plot).

- [ ] **Step 3: Commit executed notebook**

```bash
git add notebooks/analysis_rmse_vs_density.ipynb
git commit -m "feat: analysis_rmse_vs_density notebook — executed with outputs"
```

---

### Task 3: Add docs symlink + toctree entry

**Files:**
- Create: `docs/content/analysis_rmse_vs_density.ipynb` (symlink)
- Modify: `docs/index.rst`

- [ ] **Step 1: Create symlink**

```bash
ln -s ../../notebooks/analysis_rmse_vs_density.ipynb \
    docs/content/analysis_rmse_vs_density.ipynb
```

- [ ] **Step 2: Add to docs toctree**

Open `docs/index.rst` and find the `Examples` toctree block. Add the new notebook after `content/mock_example`:

```rst
   content/analysis_rmse_vs_density
```

The block should look like:

```rst
.. toctree::
   :maxdepth: 1
   :caption: Examples

   content/mock_example
   content/analysis_rmse_vs_density
```

- [ ] **Step 3: Build docs and verify**

```bash
sphinx-build docs docs/_build/html -b html -q
```

Expected: `build succeeded` (warnings for myst config lines are acceptable).

- [ ] **Step 4: Commit**

```bash
git add docs/content/analysis_rmse_vs_density.ipynb docs/index.rst
git commit -m "docs: add analysis_rmse_vs_density to examples toctree"
```

---

### Task 4: Merge to main

- [ ] **Step 1: Push branch**

```bash
git push -u origin feature/analysis-rmse
```

- [ ] **Step 2: Merge**

```bash
git checkout main
git merge --no-ff feature/analysis-rmse -m "feat: RMSE vs source density analysis notebook"
```

- [ ] **Step 3: Clean up worktree**

```bash
git worktree remove .worktrees/analysis-rmse
git branch -d feature/analysis-rmse
```

- [ ] **Step 4: Verify fast tests still pass on main**

```bash
pytest unittests/ -v -m "not slow"
```

Expected: all tests PASS.
