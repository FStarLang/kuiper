FROM kuiper-base

SHELL ["/bin/bash", "-c"]

ARG USER=ubuntu
USER $USER
ENV HOME /home/$USER
WORKDIR $HOME

# Copy kuiper (with submodules already included by build-artifact.sh)
COPY --chown=$USER kuiper/ kuiper/

# Build F*, Karamel, and Pulse so reviewers don't have to wait
RUN . $HOME/.bash_profile && cd kuiper/ && make -j$(nproc) prepare

# Install code-server and F* extension
RUN wget https://github.com/coder/code-server/releases/download/v4.111.0/code-server_4.111.0_amd64.deb \
 && sudo dpkg -i code-server*.deb \
 && rm code-server*.deb

RUN wget https://github.com/FStarLang/fstar-vscode-assistant/releases/download/v0.24.0/fstar-vscode-assistant-0.24.0.vsix \
 && code-server --install-extension fstar-vscode-assistant-*.vsix \
 && rm fstar-vscode-assistant-*.vsix

COPY entrypoint /entrypoint
ENTRYPOINT ["/entrypoint"]
