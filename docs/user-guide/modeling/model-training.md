!!! warning "Under Construction"
    This documentation is currently under active development and subject to change.
    Some sections may be incomplete or missing.

## Training Models <a id="model-training"></a>
For most of our modeling tasks, the training falls into two broad categories, the models training goal, either single patient or multi-patient, and the model type, which can be either statistical, ml, dl, or foundation based models.
### Training Goal
#### Single Patient Training
Train one model for each patient.
The goal is to create a model that minimizes its error as much as possible for a single patient.
Often it makes sense to take an existing multi-patient model and fine-tune it for a specific patient.

#### Multi-Patient Training
Train one model for all patients.
The goal is to create a generic model that works as well as possible for all patients.

### Model Type
GlucoFoundry integrates every model through a common **`ModelAdapter`** layer
(`src/models/`), built via `ModelFactory` / `create_model_and_config`, so the
training and evaluation pipeline stays model-family-agnostic. New models plug in
by implementing the `ModelAdapter` protocol. The shipped families span:

#### Statistical
Classical time-series baselines (naive, ARIMA / exponential-smoothing-style
methods) — anything that is not deep learning or a foundation model. The toolkit
exposes these through the `statistical` and `naive_baseline` families.

#### Machine Learning
Gradient-boosting and other tabular/tree-based regressors (for example the `gbm`
family).

#### Deep Learning
Neural forecasters wired through the adapter layer (for example `tsmixer`,
`patchtst`, `tide`, `tft`, and `deepar`).

#### Foundational
Pretrained time-series foundation models, fine-tuned or run zero-shot on glucose
data — for example Chronos-2, Moirai, MOMENT, Sundial, TimesFM, Toto, and
TinyTimeMixer. The toolkit's broader goal is both to adapt existing foundation
models to diabetes data and to support training diabetes foundation models from
scratch.
