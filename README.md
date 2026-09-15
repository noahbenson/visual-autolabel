# visual-autolabel: hybrid branch
Visual-area auto-labeling project repository: building a hybrid 2D/3D CNN for processing brain images.

## TODO

### Library organization (`src/`)
* We should make a clean github repo for a new slimmed-down repository: something like `mrihybridcnn` (we can think about what it should be called).
  * We'll want to copy the hybrid CNN from `visual_autolabel.image._model` into this repo, probably as a file `src/mrihybridcnn/model.py`.
  * We'll want to write a very simple dataset loader that just reads in the plain .pt files from a given 3D directory and a given 2D directory.
  * Then we can just use the training script we already have, but modify it to use the new library.
  * TL;DR—we want to simplify out all the visual-autolabel code that isn't really needed for this project and keep things simple.
  * When someone else wants to use our model, they'll need .pt files for the volume-image data and the 2D surface images; we'll provide a simple script that accepts a freesurfer subject directory and exports these data. (We can work on this last.) This will be fairly simple because all it needs to do is provide directories of the pytorch files; it doesn't need to create a special dataset object.
* Once we have final hyperparameters, we can add them to a subpackage of the library like `mrihybridcnn.li2027`, that contains the hyperparameters we found and details about training (like the `visual_autolabel.benson2025` subpackage).

### Notebooks/Documents (`notebooks/`)
* We'll need a notebook that creates all the figures for the paper—once we've made the figures and written the paper, we'll want to make sure the figures notebook runs correctly in the docker (below)
* We'll also want a documentation notebook that explains how to use the model. We'll probably write some interface functions in the `src/` library then document them as a tutorial in the doc notebook. This should explain how to use the model on a new subject, possibly how to train it with new data (that last part might be a separate notebook).

### Other Stuff
* Make sure Dockerfile and docker-compose work / can run analyses for figures (once paper is written)
* Clean up `scripts/` directory and scripts in the repo (like `new_train.py`).
* Make sure the `scripts/` README file explains how to use the scripts (if we have any).
