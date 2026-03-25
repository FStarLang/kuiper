# Base image for Kuiper CI, devcontainer, and artifact.
#
# CI / devcontainer:
#   docker build -t mtzguido/pulse-cuda-devcontainer .
#
# Artifact (see artifact/build-artifact.sh):
#   docker build -t kuiper-base .
#   docker build -t kuiper-pldi2026 -f artifact/Dockerfile artifact/

FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      wget \
      git \
      gnupg \
      sudo \
      libgmp-dev \
      opam \
      vim \
      pkg-config \
    && apt-get clean -y
# FIXME: libgmp-dev should be installed automatically by opam,
# but it is not working, so just adding it above.
# Same for pkg-config. OPAM prompts even if we're giving --yes
# and setting OPAMYES.

RUN apt-get install -y nvidia-cuda-toolkit \
 && apt-get clean -y

# Get compiled Z3
RUN wget -nv https://github.com/Z3Prover/z3/releases/download/z3-4.13.3/z3-4.13.3-x64-glibc-2.35.zip \
 && unzip z3-4.13.3-x64-glibc-2.35.zip \
 && cp z3-4.13.3-x64-glibc-2.35/bin/z3 /usr/local/bin/z3-4.13.3 \
 && rm -r z3-4.13.3-*

# Create vscode user (UID=1000) for devcontainer compatibility
ARG USER=vscode
RUN useradd -m -s /bin/bash -u 1000 $USER \
 && echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers
USER $USER
ENV HOME /home/$USER
WORKDIR $HOME

# Install OCaml
ARG OCAML_VERSION=5.4.0
RUN opam init --compiler=$OCAML_VERSION --disable-sandboxing
RUN opam update --yes
RUN opam option depext-run-installs=true
ENV OPAMYES=1
RUN opam install --yes batteries zarith stdint yojson dune menhir menhirLib pprint sedlex ppxlib process ppx_deriving ppx_deriving_yojson memtrace visitors uucp wasm fix mtime

# Instrument .profile and .bashrc to set the opam switch
RUN echo 'eval $(opam env)' | tee --append $HOME/.profile $HOME/.bashrc $HOME/.bash_profile
