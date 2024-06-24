FROM docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubi8 AS base
MAINTAINER Zane Fink <zanef2@illinois.edu>
ARG cuda_arch=70
ARG build_jobs=64
RUN \
     yum install -y dnf &&\
     dnf install -y vim git xz autoconf automake unzip python39 libcurl-devel xz-devel patch gcc-gfortran gcc-c++ bzip2 file &&\
     dnf upgrade -y &&\
     dnf clean all

RUN \
    mkdir -p hpacml-spack-env
COPY spack.yaml hpacml-spack-env/spack.yaml

FROM base AS setup-spack-env
RUN \
    git clone --depth 1 --branch releases/v0.22 https://github.com/spack/spack.git &&\
    source spack/share/spack/setup-env.sh &&\
    spack install -j64 gcc@11.4.0 && \
    spack compiler add $(spack location -i gcc@11.4.0) &&\
    spack env activate -p hpacml-spack-env &&\
    spack external find --all --exclude openssl --exclude openblas --exclude bzip2 --exclude binutils --not-buildable


FROM setup-spack-env AS install-spack-env
RUN \
    source spack/share/spack/setup-env.sh &&\
    spack env activate -p hpacml-spack-env &&\
    spack install --fail-fast --no-checksum -j64

FROM install-spack-env AS clean-spack
RUN \
    spack clean --all
