# local/ilpsolver = debian + CPLEX + Gurobi
FROM local/ilpsolver
MAINTAINER Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

RUN echo 'en_US.UTF-8 UTF-8' >/etc/locale.gen \
    && apt-get update \
    && apt-get dist-upgrade -y \
    && apt-get install -y --no-install-recommends --no-install-suggests \
        build-essential \
        cmake \
        cmake-curses-gui \
        cython3 \
        gdb \
        hdf5-helpers \
        hdf5-tools \
        less \
        libboost-all-dev \
        libgmp10-dev \
        libhdf5-dev \
        locales \
        moreutils \
        neovim \
        ninja-build \
        python3-numpy \
        tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash user
USER user
WORKDIR /home/user
COPY --chown=user:user code/ /home/user/code/
ENV CXXFLAGS=-I/usr/include/hdf5/serial/ \
    LDFLAGS=-L/usr/lib/x86_64-linux-gnu/hdf5/serial/ \
    PYTHONPATH=/home/user/code:/opt/cplex/cplex/python/3.5/x86-64_linux/ \
    PYTHONOPTIMIZE=TRUE \
    LD_LIBRARY_PATH=/home/user/build/
RUN mkdir build \
    && cd build \
    && cmake -GNinja -DCMAKE_BUILD_TYPE=Release ../code \
    && ninja
CMD ["bash", "-l"]

# vim: set ts=4 sts=4 sw=4 et:
