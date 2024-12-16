# Visual Autolabel: Hyperparameter Grid Search

This directory contains data, code, and notes on the hyperparameter grid-search
for the Visual Autolabel project. Note that the grid-search was executed on
Azure using a Docker image that has been saved to the filename
`gridsearch.tar.gz` in a Zenodo deposition with DOI
[10.5281/zenodo.14502583](https://doi.org/10.5281/zenodo.14502583).  This
repository no longer maintains the code in a state that is compatible with the
grid-search performed at that time; though the changes to the code that have
occurred since the grid search was executed are largely cosmetic and do not
contain substantive changes to the CNN model, to way that the model is trained,
or the interpretation of the dataset.  However, the docker image preserves the
code and scripts as well as the library installations and system configuration
at the time that the grid-search was run.

For more information on Docker images and how to use and interact with them, see
the documentation hosted by [Docker](https://docs.docker.com/).


## Which hyper-parameters are subjects of the grid search?

The grid search examines the following parameters:
* The base model (`base_model`), which can be either `resnet18` or
  `resnet34`. The two possible values represent ResNet models with different
  numbers of internal parameters; `resnet18` contains fewer parameters than
  `resnet34`.
* The learning rate (`lr`), which can be any of the following: $\{1.67 \times
  10^{-3}, 2.50 \times 10^{-3}, 3.75 \times 10^{-3}, 5.62 \times 10^{-3}$,
  $8.44 \times 10^{-3}\}$.
* The learning rate decay (`gamma`), which can be any of the following:
  $\{0.80, 0.85, 0.90, 0.95, 1.00\}$.
* BCE weight (`bce_weight`), which can be any of the following: $\{0.50$,
  $0.67, 0.75\}$. The BCE weight hyperparameter sets the relative weight of
  the binary cross entropy loss function relative to the dice loss function
  during the first epoch. (The overall loss function is $f(\symvec{x}) = w
  f_{\hbox{BCE}}(\symvec{x}) + (1 - w) f_{\hbox{Dice}}(\symvec{x})$ where
  $\symvec{x}$ is the vector of parameters to the loss function, $w$ is the BCE
  weight, and $f_{\hbox{BCE}}$ and $f_{\hbox{Dice}}$ are the BCE and Dice loss
  functions, respectively. During epoch 2 and 3, the BCE weight is halved, and
  during epoch 3, the BCE weight is 0.
* Batch size (`batch_size`), which can be any of the following: $\{2, 4$,
  $6\}$.


## What did we train in the grid search?

We trained 4 CNNs with each set of input parameters; these CNNs differed only in
the kinds of inputs they required and in the set of labels they produced:
1. A CNN that uses **input data from T1-weighted images alone** and predicts
   **visual area boundaries**.
2. A CNN that uses **all input data** considered in the project and predicts
   **visual area boundaries**.
3. A CNN that uses **input data from T1-weighted images alone** and predicts
   **iso-eccentric regions**.
4. A CNN that uses **all input data** considered in the project and predicts
   **iso-eccentric regions**.


## Where are the results of the grid search?

The Open Science Framework page ([osf.io/c49dv](https://osf.io/c49dv/)) for this
project includes directory in [its OSF
storage](https://osf.io/c49dv/files/osfstorage) named `hyperparameters`. In this
directory are JSON files detailing the best sets of hyperparameters from the
grid search for each of the four models above that were used during model
trainings. This directory also contains a file `grid-search.tar.gz`, which
contains the log files that document the final validation scores of all of the
individual cells in the grid search. Due to space constraints the full parameter
weights and model training data from the grid search were not retained.
