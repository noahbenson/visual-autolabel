# This Dockerfile constructs a docker image for use in training CNNs to segment
# flattened cortical surface images based on V1, V2, and V3 location.
#
# by Noah. C. Benson

# Start by using the neuropythy docker as a base.
FROM continuumio/miniconda3:4.12.0
MAINTAINER Noah C. Benson <nben@uw.edu>

# Install the needed libraries.
RUN conda install \
        pytorch torchvision torchaudio pytorch-cuda=11.8 \
        -c pytorch -c nvidia
RUN pip install \
        'neuropythy == 0.12.10' nibabel pandas torch-summary \
        matplotlib

# Go ahead and copy the library into the home directory.
RUN mkdir -p /opt/visual-autolabel
COPY . /opt/visual-autolabel

# The default method of running is to run the train script.
RUN mkdir -p /data
ENV PYTHONPATH="/opt/visual-autolabel"

# We want to run things in the data directory.
WORKDIR /data
ENTRYPOINT [ \
        "/opt/visual-autolabel/scripts/train.py", \
        "/data/opts.json", \
        "/data/train.json" \
]


