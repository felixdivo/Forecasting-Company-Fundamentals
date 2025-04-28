# syntax = docker/dockerfile:1.2

FROM mkrausio/ml_research:base-3.11-cuda-12.1.0

ARG COMPOSE_PROJ_NAME="code"

ENV PROJECT_NAME=${COMPOSE_PROJ_NAME}
RUN echo ${COMPOSE_PROJ_NAME}/$PROJECT_NAME

RUN apt-get update -y && apt-get upgrade -y && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  zip \
  unzip \
  zsh \
  git \
  git-lfs \
  ninja-build \
  python3.11-dev \
  curl \
  wget \
  cmake \
  g++-11 \
  libmkl-dev \
  && apt-get clean

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

RUN ldconfig /usr/local/cuda-12.1/compat/

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
ENV PATH=${CUDA_HOME}/bin:$PATH

RUN mkdir -p workspaces/$PROJECT_NAME

WORKDIR /workspaces/$PROJECT_NAME
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
  python -m pip install --upgrade pip && \
  pip install -r requirements.txt

CMD zsh
