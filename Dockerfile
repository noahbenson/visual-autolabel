# Dockerfile for the visual_autolabel Python library.
# 
# This Dockerfile constructs a docker image that preserves an analysis
# environment appropriate for the use of the visual_autolabel library.
# The docker image published with the paper Benson, Song, et al. (2025)
# was built using this Dockerfile.
#
# The built docker-image contains an installation of pytorch, neuropythy,
# jupyter, matplotlib, visual_autolabel, and all of their dependencies.
#
# To run the docker image, use the command:
#    docker run --rm -it nben/visual_autolabel:latest help
# 
# Author: Noah. C. Benson

# Start by using the neuropythy docker as a base.
FROM quay.io/jupyter/pytorch-notebook:pytorch-2.5.1
MAINTAINER Noah C. Benson <nben@uw.edu>

# Install the needed libraries.
RUN mamba install -c conda-forge \
        'ipyvolume >= 0.6' nibabel \
        pandas matplotlib s3fs \
        'boto3 < 1.27'
RUN pip install 'neuropythy == 0.12.16' torch-summary
# For some reason, when installing boto3 into this docker, it introduces
# a bizarre error where use of boto3 raises an AttributeError about a
# failure to find boto3.utils, which is imported at the top of the
# boto3.session file and used later, but the attribute error implies that
# somehow the module has lost track of it. Regardless, we can hack a fix
# into the module, which is needed to get AWS credentials out of the
# ~/.aws/credentials file.
RUN SESFL="$(python -c 'import boto3.session; print(boto3.session.__file__);')" \
  ; cp "$SESFL" /home/jovyan/session.py \
 && cat /home/jovyan/session.py \
  | sed -E 's/import boto3\.utils/from . import utils as _bugfix_boto3_utils/g' \
  | sed -E 's/boto3\.utils/_bugfix_boto3_utils/g' > "$SESFL" \
 && rm /home/jovyan/session.py

# Install FreeSurfer's fsaverage subject.
USER root
RUN mkdir -p /coredata/fssubjects \
 && curl -L -o /coredata/fssubjects/fsaverage.tar.gz \
      https://github.com/noahbenson/neuropythy/wiki/files/fsaverage.tar.gz \
 && cd /coredata/fssubjects \
 && tar zxf fsaverage.tar.gz \
 && rm fsaverage.tar.gz
USER $NB_USER

# Disable the jupyterlab notifications extension.
RUN jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

# Warm up the python libraries.
RUN python -c 'import neuropythy, boto3, matplotlib, matplotlib.pyplot, pandas'


# A few final things should be done as root:
USER root

# We want to run things in the /root (home) directory, and we want a /data
# directory to mount things into.
RUN mkdir -p /data \
             /data/visual-autolabel \
             /data/hcp /data/hcp/meta \
             /data/performance-fields

# We want to pre-install the model weights for the areas and rings.
RUN mkdir -p /data/visual-autolabel/models/benson2025/anat_area
RUN curl -o /data/visual-autolabel/models/benson2025/anat_area/model.pt \
        https://osf.io/download/67636e408dce3a7450a345b8/
RUN curl -o /data/visual-autolabel/models/benson2025/anat_area/options.json \
        https://osf.io/download/67636e69094e7a3c4db71d63/
RUN curl -o /data/visual-autolabel/models/benson2025/anat_area/plan.json \
        https://osf.io/download/67636e5fe5ffed714fcf0aa2/
RUN curl -o /data/visual-autolabel/models/benson2025/anat_area/training.tsv \
        https://osf.io/download/67636e59baeaf4d3becf0e85/
RUN mkdir -p /data/visual-autolabel/models/benson2025/anat_ring
#RUN curl -o /data/visual-autolabel/models/benson2025/anat_ring/model.pt \
#
#RUN curl -o /data/visual-autolabel/models/benson2025/anat_ring/options.json \
#
#RUN curl -o /data/visual-autolabel/models/benson2025/anat_ring/plan.json \
#
#RUN curl -o /data/visual-autolabel/models/benson2025/anat_ring/training.tsv \
#

# Configure neuropythy:
COPY docker/npythyrc.json /home/jovyan/.npythyrc

# Copy our notebook in:
COPY benson2025/analysis.ipynb /home/jovyan/benson2025-analysis.ipynb
# Fix the permissions on the analysis notebook.
RUN fix-permissions "${HOME}/benson2025-analysis.ipynb"

# Put the startup script somewhere.
COPY docker/startup.sh /startup.sh
RUN chown root:root /startup.sh && chmod 755 /startup.sh

# Put the library into the image:
COPY . $HOME/visual-autolabel
RUN fix-permissions "${HOME}/visual-autolabel"

# Switch back to the notebook user for the end.
USER $NB_USER

# Install the library
RUN cd "$HOME/visual-autolabel" \
 && pip install --no-cache-dir .

ENTRYPOINT ["tini", "-g", "--", "/startup.sh"]
