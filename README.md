
# `visual-autolabel`: Segmenting the Human Visual Cortex using Convolutional Neural Networks

This GitHub repository contains the `visual_autolabel` python library, which
uses the [Young Adult Human Connectome Project (HCP)](
https://db.humanconnectome.org/), and the [HCP 7 Tesla Retinotopy Dataset](
https://doi.org/10.1167/18.13.23) to train convolutional neural networks to
predict both the boundaries of visual area V1, V2, and V3 and a set of five
orthogonal quasi-iso-eccentric regions spanning 0&ndash;0.5&deg;,
0.5&ndash;1&deg;, 1&ndash;2&deg;, 2&ndash;4&deg;, and 4&ndash;7&deg; of visual
eccentricity within the same areas.


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
 * `Dockerfile`. This file contains instructions to build a docker image that
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
* `-b` or `--both`. Predict both the visual areas and the 5 iso-eccentric
  regions.
* `-V <template>` or `--volume=<template>`. Also output a Nifti or MGZ volume,
  depending on the file type of the given template image, of the visual areas or
  rings. The file is written to the subject's `mri/` directory by default. The
  template image must be an image file itself or the string `'native'` or
  `'raw'`. If an image file (`.mgz`, `.mgh`, `.nii`, or `.nii.gz`) is given,
  then the resulting image will have the same affine, shape, and file type as
  the template. The option `'native'` results in an image with the native
  FreeSurfer orientation, such as in the subject's `ribbon.mgz` file while the
  option `'raw'` results in an image with the subjects' `rawavg.mgz` file, which
  is typically aligned to the scanner.
* `-a` or `--annot`. Also output a FreeSurfer annotation file, by default to the
  subject's `label/` directory.
* `-x` or `--no-surface`. Skip writing out the surface MGZ files to the
  subject's `surf/` by default.
* `-o` or `--output-dir=<dir>`. Write the output files to the given directory
  instead of to the subject's subdirectories.
* `-t<value>` or `--tag=<value>`. Change the tag that is used in output
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
>        --both \
>        --tag='{subject}.benson2025' \
>        s3://natural-scenes-dataset/nsddata/freesurfer/subj01 \
>        s3://natural-scenes-dataset/nsddata/freesurfer/subj02 \
>        s3://natural-scenes-dataset/nsddata/freesurfer/subj03

# (The above command will take several minutes to run.)

$ ls
lh.subj01.benson2025_varea.mgz
lh.subj01.benson2025_vring.mgz
lh.subj02.benson2025_varea.mgz
lh.subj02.benson2025_vring.mgz
lh.subj03.benson2025_varea.mgz
lh.subj03.benson2025_vring.mgz
rh.subj01.benson2025_varea.mgz
rh.subj01.benson2025_vring.mgz
rh.subj02.benson2025_varea.mgz
rh.subj02.benson2025_vring.mgz
rh.subj03.benson2025_varea.mgz
rh.subj03.benson2025_vring.mgz
```

### Output Files

The outputs produced by the files mark vertices (or voxels in the case of
volumetric image outputs) with one of the following labels:

**Visual Areas (`*_varea.*`)**  
| Label | Area |
|:-----:|:----:|
|  1    |  V1  |
|  2    |  V2  |
|  3    |  V3  |
|  0    | none |

**Iso-eccentric Rings (`*_vring.*`)**
| Label | Ring                  |
|:-----:|:---------------------:|
|  1    | 0&deg;&ndash;0.5&deg; |
|  2    | 0.5&deg;&ndash;1&deg; |
|  3    | 1&deg;&ndash;2&deg;   |
|  4    | 2&deg;&ndash;4&deg;   |
|  5    | 4&deg;&ndash;7&deg;   |
|  0    | none                  |


### Limitations

The CNNs applied by the `visual-autolabel` library use as input data a variety
of anatomical features processed by FreeSurfer, specifically the curvature, the
sulcal depth, the gray matter thickness, and the vertex surface area. There are
a number of requirements for this to work and the outputs have a number of
caveats, all of which are listed here.

**Requirements**
* The subject provided must have been processed by FreeSurfer (i.e., the input
  directory must be a FreeSurfer subject directory).
* The subject must have been registered to FreeSurfer's *fsaverage* subject.

**Caveats**
* The predictions made by the CNNs are for the most foveal parts of V1, V2, and
  V3 only: they should cover 0&deg;&ndash;7&deg; of eccentricity.
* The CNNs have been both cross-validated on a left-out dataset and evaluated on
  an independent dataset. Combined, the accuracy of the models was 74%, averaged
  over V1, V2, and V3, using the S&oslash;rensen-Dice coefficient with only
  small differences in accuracy between the dataset. In V1, V2, and V3
  separately, the accuracy was approximately 84%, 74%, and 63%. For comparison,
  the inter-rater reliability of human experts who draw the boundaries on the
  functional maps by hand was 93%, 88% and 83%, respectively.
* Because the models are CNNs, it is likely that some subjects will be
  dissimilar enough from the original training dataset that the predictions will
  be unreliable. We suggest always checking the predictions by hand before using
  them in analyses. In particular, we do not expect this model to perform well
  on subjects with brain abnormalities or atypical brain structure.


## Acknowledgements

This work was funded by NEI grant 1R01EY033628 (to N.C.B).

