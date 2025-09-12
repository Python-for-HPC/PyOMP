ARG UBUNTU_VER=22.04

FROM ubuntu:${UBUNTU_VER} AS base
RUN \
  apt-get -qq update > /dev/null &&\
  apt-get -qq upgrade > /dev/null &&\
  apt-get -qq install wget neovim > /dev/null &&\
  useradd -ms /bin/bash pyompuser

USER pyompuser
WORKDIR /home/pyompuser
COPY --chown=pyompuser examples /home/pyompuser/examples
SHELL ["/bin/bash", "-lc"]
RUN \
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O miniconda.sh &&\
  /bin/bash miniconda.sh -b -u -p ./miniconda3 &&\
  rm -rf miniconda.sh &&\
  miniconda3/bin/conda init

RUN \
  source /home/pyompuser/miniconda3/bin/activate &&\
  export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true &&\
  conda create -y -n pyomp -c python-for-hpc -c conda-forge \
    pyomp jupyter notebook python=3.10 &&\
  echo "conda activate pyomp" >> /home/pyompuser/.bashrc

CMD /bin/bash -c "source /home/pyompuser/miniconda3/bin/activate && \
  conda activate pyomp && \
  jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser & tail -f /dev/null"
