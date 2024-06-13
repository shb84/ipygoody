# ipymodelview

A library of Jupyter widgets to visualize any vectorized model with a signature `y = f(x)` where `x, y` are arrays. For example, 
`f(x)` could be a surrogate model (e.g. response surface equation or neural network) or it could be a physics-based model (e.g. an `openmdao` model). 

--- 
## Installation

#### `dev`
Assuming [conda](https://conda.org/) is installed:

```bash
conda env update --file environment.yml --prefix ./venv
conda activate ./venv
pip install -e . 
pytest
```

#### `ci`

Commands are run with [`doit`](https://pydoit.org/). To see all `doit` commands:

```bash
doit list --all --status
```

To install the `ci` environment: 

```bash
doit env install test
```

---
## Contribution

The recommended process is to do all work in the `dev` environment. 
Upon satisfaction, before merge requests, follow the steps below.  

#### Step 1: Update Environment Specs (optional)

_**IF AND ONLY IF** the `environment.yml` file was updated during during development, then the `ci` environment must also be updated accordingly. To do so, update `deploy/specs/run.yml` and re-generate the lock files_: 
 
```bash
doit lock
```

#### Step 2: Run Unit Tests

_Make sure the unit tests are passing_: 

```bash
doit install test
```

#### Step 3: Fix Lint Issues 

_Make sure the code is well formatted. Fix manually if needed_: 

```bash
doit fix lint
```

#### Step 4: Run Notebooks (optional) 

_If applicable, mannually check notebooks in `ci` environment_: 

```bash
doit lab
```

--- 
## Documentation 

Documentation is available [here](TODO) (generated using `sphinx`)

--- 
## Usage
TODO

--- 
## License
Distributed under the terms of the Modified BSD License.
