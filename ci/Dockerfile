FROM mtzguido/pulse-base-devcontainer

USER root

RUN apt-get install -y nvidia-cuda-toolkit \
 && apt-get clean -y
 
USER vscode
