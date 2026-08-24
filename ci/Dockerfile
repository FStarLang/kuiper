# Base image for Kuiper CI, devcontainer, and artifact.
#
# Two stages are published:
#
#   nocuda  everything except the CUDA toolkit. Enough to verify and
#           extract, which is all the jobs on GitHub-hosted runners need; nvcc
#           is neither available nor useful there. Published as
#           ghcr.io/fstarlang/kuiper-base-nocuda. ~0.6GB compressed.
#   cuda    nocuda plus nvidia-cuda-toolkit, for the jobs on the GPU runner,
#           the devcontainer, and the artifact. Published as
#           ghcr.io/fstarlang/kuiper-base. ~4.2GB compressed.
#
# CI publishes both to ghcr.io, see .github/workflows/container.yml. To build
# and push them by hand, see ci/rebuild-container-and-deploy.sh. To build one
# locally:
#
#   docker build --target nocuda -t kuiper-base-nocuda -f ci/Dockerfile ci/
#   docker build --target cuda   -t kuiper-base        -f ci/Dockerfile ci/
#
# Artifact (see artifact/build-artifact.sh):
#   docker build --target cuda -t kuiper-base -f ../ci/Dockerfile ../ci/
#   docker build -t kuiper-pldi2026 -f artifact/Dockerfile artifact/

FROM ubuntu:24.04 AS nocuda

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      wget \
      git \
      gnupg \
      indent \
      sudo \
      libgmp-dev \
      opam \
      python3 \
      unzip \
      vim \
      time \
      pkg-config \
    && apt-get clean -y
# FIXME: libgmp-dev should be installed automatically by opam,
# but it is not working, so just adding it above.
# Same for pkg-config. OPAM prompts even if we're giving --yes
# and setting OPAMYES.

# Get compiled Z3
RUN wget -nv https://github.com/Z3Prover/z3/releases/download/z3-4.13.3/z3-4.13.3-x64-glibc-2.35.zip \
 && unzip z3-4.13.3-x64-glibc-2.35.zip \
 && cp z3-4.13.3-x64-glibc-2.35/bin/z3 /usr/local/bin/z3-4.13.3 \
 && rm -r z3-4.13.3-*

# Create a new user and give them sudo rights
ARG USER=vscode
RUN useradd -d /home/$USER -s /bin/bash -m $USER
RUN echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers
USER $USER
ENV HOME /home/$USER

# Install OCaml
ARG OCAML_VERSION=5.4.0
RUN opam init --compiler=$OCAML_VERSION --disable-sandboxing
RUN opam update --yes
RUN opam option depext-run-installs=true
ENV OPAMYES=1
RUN opam install --yes batteries zarith stdint yojson dune menhir menhirLib pprint sedlex ppxlib process ppx_deriving ppx_deriving_yojson memtrace visitors uucp wasm fix mtime

# Instrument .profile and .bashrc to set the opam switch
RUN echo 'eval $(opam env)' | tee --append $HOME/.profile $HOME/.bashrc $HOME/.bash_profile

# The CUDA toolkit is over 3.5GB of the 4.2GB image, and is only needed to
# compile and run the generated kernels, never to verify or extract them.
# Keeping it in its own stage lets the cloud jobs pull the nocuda image.
FROM nocuda AS cuda

USER root
RUN apt-get update \
 && apt-get install -y nvidia-cuda-toolkit \
 && apt-get clean -y
ARG USER=vscode
USER $USER
