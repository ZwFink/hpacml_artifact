FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-devel AS base
MAINTAINER Zane Fink <zanef2@illinois.edu>
ARG cuda_arch=70
ARG build_jobs=64

RUN \
    apt-get update &&\
    apt-get install -y vim git xz-utils autoconf automake libssl-dev unzip libcurl4-openssl-dev xz-utils patch bzip2 gfortran file &&\
    apt-get upgrade -y &&\
    apt-get clean

RUN \
    mkdir -p hpacml-spack-env
COPY spack.yaml hpacml-spack-env/spack.yaml

FROM base AS setup-spack-env
RUN \
    git clone --depth 1 --branch releases/v0.22 https://github.com/spack/spack.git &&\
    . spack/share/spack/setup-env.sh &&\
    spack compiler find &&\
    spack env activate -p hpacml-spack-env &&\
    spack external find --all --exclude openssl --exclude openblas --exclude bzip2 --exclude binutils --not-buildable


FROM setup-spack-env AS install-spack-env
RUN \
    . spack/share/spack/setup-env.sh &&\
    spack env activate -p hpacml-spack-env &&\
    spack install --fail-fast --no-checksum -j2

FROM install-spack-env AS clean-spack
RUN \
    spack clean --all
