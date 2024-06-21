FROM docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubi8 AS base
MAINTAINER Zane Fink <zanef2@illinois.edu>
ARG cuda_arch=70
ARG build_jobs=36
RUN \
     yum install -y dnf &&\
     dnf install -y vim git xz autoconf automake unzip patch gcc-gfortran gcc-c++ bzip2 file &&\
     dnf upgrade -y &&\
     dnf clean all

RUN \
    mkdir -p hpacml-spack-env
COPY spack.yaml hpacml-spack-env/spack.yaml

FROM base AS setup-spack-env
RUN \
    git clone --depth 1 --branch releases/v0.22 https://github.com/spack/spack.git &&\
    source spack/share/spack/setup-env.sh &&\
    spack install -j36 gcc@11.4.0 &&\
    spack env activate -p hpacml-spack-env &&\
    spack external find --all --not-buildable --exclude openssl --exclude openblas --exclude bzip2


FROM setup-spack-env AS install-spack-env
RUN \
    source spack/share/spack/setup-env.sh &&\
    spack env activate -p hpacml-spack-env &&\
    spack install --fail-fast -j36

FROM install-spack-env AS clean-spack
RUN \
    spack clean --all
