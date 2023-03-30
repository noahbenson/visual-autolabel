# Visual Autolabel Grid Search

This file contains notes on the grid-search for the Visual Autolabel project.

## What parameters are we using in the grid search?

Possibilities for parameters:
* The base model (`base_model`: `resnet18` or `resnet34`)
* Whether the resnet is pretrained (`pretrained`)
* The learning rate (`lr`; we have been using `0.00125-0.00375`)  
  (Bogeng is doing some reading on what is typical for these parameters).
* The learning rate decay (`gamma`)  
  (Bogeng is doing some reading on what is typical for these parameters).
* BCE weight (`bce_weight`; just enough to figure out whether our values matter or not). This may be hard to find information about in the literature: we are changing the weighting of two different loss functions (in the beginning we prefer the BCE and at the end we prefer the dice loss).
  (Bogeng is doing some reading on what is typical for these parameters).
* Batch size (`batch_size`; I believe we're using 5). Probably it would be sufficient to use values `[2, 3, 4, 5, 6, 7, 8, 9, 10]` or maybe only the odd values.
* This may be as many as 1800 cells!  
  (Noah will look into how much this would actually cost on AWS and how many
  cells are feasible.)


## What do we train in the grid search?

* Basically two kinds results we get when we train the CNN:
  * First, if the training data has functional data (polar angle, eccentricity) in it, we get strong results (similar to humans).
  * Second, if we give it anything else but functional data (anatomical or diffusion data) it does less well but still better than previous (non-CNN) methods.
* So we should run the grid search over two different sets of inputs: (1) everything including functional data, and (2) everything except functional data.


## How do we run the script / training?

* We already have a script in the `scripts` directory of this repository that runs training for a model given 2 JSON files (one for the training options and one for the training plan).
* All of the options can be in the plan instead of in the options if needed.
* Would be nice to have, at the end of the grid search, 1 directory per "cell" in the grid (a cell being a single set of parameters for training a single model), with one JSON file of the input options in the directory, and the results of the model training. These directories can be in this `grid-search` directory.
* We can set up a `Dockerfile` that makes a docker image that can run the training for a single model: the docker image should take one of the grid-search cell directories as input and should write the results into that directory.
* Then, all we'll need to do is run this docker image once for each grid-search cell on AWS.

