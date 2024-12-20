
# `visual-autolabel`: Segmenting the Human Visual Cortex using Convolutional Neural Networks

This GitHub repository contains the `visual_autolabel` python library, which
uses the [Young Adult Human Connectome Project (HCP)](
https://db.humanconnectome.org/), and the [HCP 7 Tesla Retinotopy Dataset](
https://doi.org/10.1167/18.13.23) to train convolutional neural networks to
predict both the boundaries of visual area V1, V2, and V3 and a set of five
orthogonal quasi-iso-eccentric regions spanning 0-0.5, 0.5-1, 1-2, 2-4, and 4-7
degrees of visual eccentricity within the same areas.


## Authors

* **[Noah C. Benson](mailto:nben@uw.edu)** (*Corresponding Author*)  
  eScience Institute  
  University of Washington  
  Seattle, United States
* **Bogeng Song**  
  Department of Psychology  
  New York University  
  New York, United States
  
  *Current Affiliation*  
  Department of Psychology  
  Georgia Institute of Technology  
  Georgia, United States
* **Shaoling Chen**  
  Courant Institute  
  New York University  
  New York, United States
  
  _*Current Affiliation*_  
  Amazon.com, Inc.  
  Seattle, United States


## Repository Contents

This repository contains the following files and directories:
 * `benson2025/`. This folder contains non-code metadata related to the paper
   by Benson, Song, et al. (2025). This includes information about the training
   regime used in that paper, information about the hyperparameter search, and
   the analysis notebook that produced the figures used in the paper.
 * `docker/`. A folder that contains a configuration files and scripts for use
   in the Docker image that can be built from this repository. The files in
   this directory are used by the `Dockerfile` when building this image.
 * `src/visual_autolabel/`. This folder contains the Python code for the
   `visual_autolabel` library. This library is both a toolkit that can be
   extended to rapidly create cached datasets for training new CNNs from
   cortical surface data and a set of documentation and code for the paper
   associated with this repository.
 * `Dockerfile`. This file contains instructions to build a docker image that is
   stores an environment equivalent to the environment in which the analyses and
   figures of the original paper were generated.
 * `LICENSE`. This project has been released under the MIT license.
 * `README.md`. This README file.
 * `docker-compose.yml`. This docker-compose file allows one to run
   `docker-compose up` to start a Jupyter server inside of a docker container
   that replicates the environment in which the original analyses were
   performed.


## Applying the CNNs to a Subject

The library works with arbitrary FreeSurfer subjects and can be used from the
command-line to apply the CNNs trained by Benson, Song, et al. (2025) to such
subjects. This can be done in two ways. The first, recommended, way is to use
`docker` to run the model. If you have `docker` installed and the Docker daemon
or Docker Desktop running in the background, then in a terminal, you can use the
following command:

```bash
docker --rm -it -v /my/freesufer/subjects:/subjects \
       nben/benson2025-unet:latest --verbose bert
```

Alternatively, you can install the library in this repository then call it
directly:

```bash
python -m visual_autolabel --verbose /my/freesurfer/subjects/bert
```

Both of the above methods evaluate the models on subject `bert` in the directory
`/my/freesurfer/subjects` and write out label files in `bert`'s subject
directory containing the predicted V1, V2, and V3 labels. The behavior of the
command can be changed by a few options:

* `-v` or `--verbose`. Print status messages.
* `-r` or `--rings`. Predict the 5 iso-eccentric regions instead of the visual
  area boundaries.
* `-V <template>` or `--volume=<template>`. Also output a Nifti or MGZ volume,
  depending on the file type of the given template image, of the visual areas or
  rings. The file is written to the subject's `mri/` directory by default.
* `-a` or `--annot`. Also output a FreeSurfer annotation file, by default to the
  subject's `label/` directory.
* `-x` or `--no-surface`. Skip writing out the surface MGZ files to the
  subject's `surf/` by default.
* `-o` or `--output-dir=<dir>`. Write the output files to the given directory
  instead of to the subject's subdirectories.
* `-t<value>` or `--tag=<value>`. Change the tag that is use in output
  filenames. The filenames used by `visual_autolabel` are, for surface MGZ and
  annotation file outputs, `{hemisphere}.{tag}_{label}.{ext}`, and for
  volumetric images, `{tag}_{label}.{ext}`. The default `tag` is `'benson2025'`,
  and `hemisphere`, `label`, and `ext` are one of `{'lh', 'rh'}`, `{'varea',
  'vring'}`, and `{'mgz', 'nii.gz', 'annot'}`, respectively. By changing the tag
  you can change the filename. You may additionally use Python format string
  `{subject}` in the tag.

Note that the subject path can optionally be an `s3` path as long as a writeable
output directory is given, so the following example applies the model to the
first three subjects of the [Natural Scenes Dataset]( 
https://naturalscenesdataset.org/):

```bash
$ docker --rm -it -v .:/out \
>        nben/benson2025-unet:latest \
>        --output-dir=/out \
>        --tag='{subject}.benson2025' \
>        s3://natural-scenes-dataset/nsddata/freesurfer/subj01 \
>        s3://natural-scenes-dataset/nsddata/freesurfer/subj02 \
>        s3://natural-scenes-dataset/nsddata/freesurfer/subj03

# (The above command will take several minutes to run.)

$ ls
lh.subj01.benson2025_varea.mgz
lh.subj02.benson2025_varea.mgz
lh.subj03.benson2025_varea.mgz
rh.subj01.benson2025_varea.mgz
rh.subj02.benson2025_varea.mgz
rh.subj03.benson2025_varea.mgz
```

## Acknowledgements

This work was funded by NEI grant 1R01EY033628 (to N.C.B).

