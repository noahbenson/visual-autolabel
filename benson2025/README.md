# Publication-associated Metadata

This directory contains various metadata and documents associated with the
publication by Benson, Song, et al. (2025) that is itself associated with the
`visual-autolabel` repository. The contents of this directory are documented
below.


## `hyperparameters/`

The `hyperparameters` directory contains the instructions originally used to
build the `grid-search` virtual machine. The virtual machine itself has been
preserved and published
(DOI:[10.5281/zenodo.14502583](https://doi.org/10.5281/zenodo.14502583)), but
the code and instructions are largely included as an artifact, as the repository
is no longer in a state that is usable by the original grid-search scripts
(though the changes that render the code unusable generally regard its
organization rather than its substance).


## `training-regimes/`

This directory contains the training regime instructions derived from the
hyperparameter search. These are stored as JSON files containing options and
epoch execution plans, which also serve as example execution plans for the
`visual_autolabel` library.


## `analysis.ipynb`

This notebook contains documentation, analyses, and visualizations that were
performed by Benson, Song, et al. (2025) in their report on the segmentation of
V1, V2, and V3 using convolutional neural networks. It was published together
with a virtual machine, containerized in a docker image, with the intention of
providing a persistent means of reproducing the computation in the original
paper. To duplicate the environment used in the analysis of the publication, the
following instructions are provided:

1. **Obtain access to the Human Connectome Project (HCP) data**.
    1. **Register at the [HCP connectome database page](https://db.humanconnectome.org/)**.
    2. **Obtain and save AWS access credentials**. Once you have an account, log
       into the database; near the top of the initial splash page is a cell
       titled "WU-Minn HCP Data - 1200 Subjects", and inside this cell is a
       button for activating Amazon S3 Access. When you activate this feature,
       you will be given a "Key" and a "Secret". These should be saved to the
       file `${HOME}/.aws/credentials` under the heading `[hcp]` where `${HOME}`
       is your home directory. For more information about the format of this
       file, see [this
       page](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-files.html),
       but, in brief: if you don't already have a credentials file, you can put
       the following block of text in it, and if you do already have such a
       file, you can append the following text to the end of it except with the
       `______________` replaced with your key and the
       `********************` replaced with your secret.  
       ```
       [hcp]
       aws_access_key_id = ______________
       aws_secret_access_key = ********************
       ```
    3. **Obtain access to the restricted dataset and save the restricted data
       file**. To obtain access to the restricted data, you need to sign an
       agreement and send it to the HCP. For more information on restricted data
       access, see [this
       page](https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage). Once
       your access has been granted, you should receive instructions on
       obtaining the restricted data CSV file.
2. **Set up Docker and obtain the docker image**.
    1. **Install and start [docker](https://docker.com/)'s `docker-desktop`
       service**. See [this
       page](https://docs.docker.com/get-started/get-docker/) for download and
       installation instructions. Note that docker requires administrator
       privileges.
    2. **Download the `analysis.tar.gz` docker image from the repository**
       (DOI:[10.5281/zenodo.14502583](https://doi.org/10.5281/zenodo.14502583)).
    3. **Unzip the analysis image file**.  
       ```bash
       gunzip analyziz.tar.gz
       ```
    4. **Load the docker image**.  
       ```bash
       docker load -i analysis.tar
       ```
3. **Use the `docker run` command to start the docker image**. This command
   should be structured as follows with the exception that text in
   angle-brackets (`<text>`), including the brackets, should be replaced with an
   appropriate substitution for the local machine on which you are running the
   virtual machine.  
   ```bash
   docker run --rm -it \
              -p 8888:8888 \
              -v <HCP-restricted-data>:/data/hcp/meta/RESTRICTED_full.csv \
              -v "${HOME}/.aws:/home/jovyan/.aws" \
              nben/visual_autolabel:benson2025 jupyter
   ```

The above command will start a Jupyter server inside the virtual machine and
will expose it on port 8888, allowing you to point a browser to
`http://127.0.0.1:8888/` to connect to the virtual machine's compute
environment. This environment should remain identical to that used to analyze
the data for the paper. The parameter `<HCP-restricted-data>` in the command is
the path on your local environment to the restricted data CSV file from the
Human Connectome Project (see instruction 1C, above).
                                                                                                                                              
You can optionally include additional volume mount data in order to save cached
data across uses of the docker image or to reduce compute time. If, for example,
you have the HCP subject data from the 1200 subject release loaded in the
directory `/hcp/subjects` on your local computer, you can add the line `-v
/hcp/subjects:/data/hcp/subjects` after the `-p 8888:8888` line. If you wish to
save cache files across runs so that subsequent computations are faster, you can
use put the line `-v <cache-dir>:/data` after the `-p 8888:8888` line where
`<cache-dir>` should be replaced with a directory that will hold all the various
cache data.

