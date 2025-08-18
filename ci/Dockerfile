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

# RUN apt-get install -y nvidia-cuda-toolkit \
#  && apt-get clean -y

RUN wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run
RUN sudo apt-get -y install libxml2 && apt-get clean -y # apparently needed to run the runfile
RUN sudo bash ./cuda_13.0.0_580.65.06_linux.run --silent --toolkit # No driver, this is just a container

# Get compiled Z3
RUN wget -nv https://github.com/Z3Prover/z3/releases/download/Z3-4.8.5/z3-4.8.5-x64-ubuntu-16.04.zip \
 && unzip z3-4.8.5-x64-ubuntu-16.04.zip \
 && cp z3-4.8.5-x64-ubuntu-16.04/bin/z3 /usr/local/bin/z3-4.8.5 \
 && rm -r z3-4.8.5-*

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

RUN echo 'PATH="/usr/local/cuda/bin:$PATH"' >> /home/vscode/.profile
RUN echo 'PATH="/usr/local/cuda/bin:$PATH"' >> /home/vscode/.bashrc
RUN echo 'PATH="/usr/local/cuda/bin:$PATH"' >> /home/vscode/.bash_profile

# Install OCaml
ARG OCAML_VERSION=5.3.0
RUN opam init --compiler=$OCAML_VERSION --disable-sandboxing
RUN opam update --yes
RUN opam option depext-run-installs=true
ENV OPAMYES=1
RUN opam install --yes batteries zarith stdint yojson dune menhir menhirLib pprint sedlex ppxlib process ppx_deriving ppx_deriving_yojson memtrace visitors uucp wasm fix mtime
