# visual-autolabel: hybrid branch
Visual-area auto-labeling project repository: building a hybrid 2D/3D CNN for processing brain images.

## TODO

### Library organization (`src/`)
* Once we have final hyperparameters, we'll need to add a new subpackage, `visual_autolabel.li2027`, that contains the hyperparameters we found and details about training (like the `visual_autolabel.benson2025` subpackage).
* We'll need to clean up some of the old code (like `Image3DTo2DDataset`) in the image namespace.
* We'll need to add a command for applying the model (in the `visual_autolabel.cmd` namespace); the current `visual_autolabel.cmd._apply` namespace contains code for loading a single FreeSurfer subject's 2D data already; we'll need to add the ability to load 3D data (and this might be tricky, see below).
* The `_data3D.py` dataset file contains classes for loading HCP data. We'll need similar classes for loading data directly from a FreeSurfer subject directory (i.e., people will want to apply this model to their subjects not just HCP subjects).
  * We'll need to write a new class like the `HCPDataset3D`, but `FreeSurferDataset3D` that loads a single sample from a FreeSurfer subject directory. (It will mostly be a copy of `HCPDataset3D`.)
  * We'll probably need to align any FreeSurfer brain to MNI using a rigid-body transform (or we might just document how to do this and require them to provide the affine)
  * Getting diffusion data to load in may be challenging because there isn't a standard format that we know well; let's revisit once we have final results.

### Notebooks/Documents (`notebooks/`)
* We'll need a notebook that creates all the figures for the paper—once we've made the figures and written the paper, we'll want to make sure the figures notebook runs correctly in the docker (below)
* We'll also want a documentation notebook that explains how to use the model. We'll probably write some interface functions in the `src/` library then document them as a tutorial in the doc notebook. This should explain how to use the model on a new subject, possibly how to train it with new data (that last part might be a separate notebook).

### Other Stuff
* Make sure Dockerfile and docker-compose work / can run analyses for figures (once paper is written)
* Clean up `scripts/` directory and scripts in the repo (like `new_train.py`).
* Make sure the `scripts/` README file explains how to use the scripts (if we have any).
