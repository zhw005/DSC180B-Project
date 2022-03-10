# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
# ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.2-stable

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
ARG BASE_CONTAINER=ucsdets/datascience-notebook:2021.2-stable

# scipy/machine learning (tensorflow, pytorch)
# https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
# ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2021.3-42158c8

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN echo 'jupyter notebook "$@"' > /run_jupyter.sh && chmod 755 /run_jupyter.sh

# RUN apt-get -y install aria2 nmap traceroute

# 3) install packages using notebook user
USER jovyan

# RUN conda install -y scikit-learn

RUN pip install --no-cache-dir torch==1.10.0
RUN pip install --no-cache-dir torchtext==0.11.0
RUN pip install --no-cache-dir shap==0.40.0
RUN pip install --no-cache-dir numpy==1.19.5
RUN pip install --no-cache-dir lime==0.2.0.1
RUN pip install --no-cache-dir scipy
RUN pip install --no-cache-dir aix360
RUN pip install --no-cache-dir pydot==1.3.0
RUN pip install --no-cache-dir dowhy==0.6
RUN pip install --no-cache-dir scikit-learn==1.0.1
RUN pip install --no-cache-dir lightgbm
RUN pip install --no-cache-dir xgboost
RUN pip install --no-cache-dir pytorch_tabnet
RUN pip install --no-cache-dir mip

# Override command to disable running jupyter notebook at launch
#CMD ["/bin/bash"]
