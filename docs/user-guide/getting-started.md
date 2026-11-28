!!! warning "Under Construction"
    This documentation is currently under active development and subject to change.
    Some sections may be incomplete or missing.

GlucoFoundry is a task-general, reusable toolkit for glucose time-series
foundation-model research. You fork it to run an experiment — forecasting,
benchmarking, fine-tuning or training foundation models, event detection, and
more — while the reusable core (data loaders, the model-adapter layer, the Optuna
sweep framework, evaluation, and workflows) stays shared. See the
[toolkit architecture](../architecture/toolkit-architecture.md) for the full
design, and the shipped `experiments/nocturnal/` recipe (overnight,
midnight-anchored hypoglycemia forecasting) for a worked example.

## Setting up your project

### Local repo
This is just creates a local version of this repository so you can start making your own changes
1. In your local terminal, navigate to your desired project directory.
2. Run the following command: `git clone https://github.com/Blood-Glucose-Control/GlucoFoundry.git`

### Local environment
This ensures that everyone is working with the same tools and that the code you write will run on everyone else's machine, including our model training servers.

3. Create a virtual environment with python 3.12 in your terminal in the projects root directory

    `python3.12 -m venv .venv`

    - **NOTE:** If you're using a different .venv name, please ensure it's added to the .gitignore file and not pushed to our repo.
4. Activate the venv:
    **Ubuntu:** `source .venv/bin/activate`
    **Windows:** `.venv\Scripts\Activate.ps1`
  - Your terminal should now show the venv being active:
    - `(.venv) user@host:~/GlucoFoundry$ `
5. Install the toolkit (editable) together with its dependencies:

    `pip install -e .`

    The repo is pip-installable — everything is importable from `src/`, so avoid
    local file paths. Re-run this after pulling changes that touch dependencies;
    the first run takes the longest. Read more about editable installs
    [here](https://goodresearch.dev/setup#create-a-pip-installable-package-recommended).

Congratulations! You're ready to start contributing to our project.

---

## Pull Requests
Please run pre-commit and resolve any errors before making a PR.
### Run pre-commit
`pre-commit run --all-files`

### Testing
`pytest --ignore=tests/models` (model-family tests run via `make test-<family>`)


## Training Models
Please see our [model training section](modeling/model-training.md#model-training) for advanced usage.

For runnable entrypoints, see the [example scripts guide](example-scripts-guide.md).

## Running Benchmarks
Please see our [model benchmarking section](benchmarking/model-benchmarking.md#model-benchmarking) for advanced usage.

For runnable entrypoints, see the [example scripts guide](example-scripts-guide.md).
