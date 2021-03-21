FROM dolfinx/dev-env

ARG DOLFINY_BUILD_TYPE=Release

ARG UFL_GIT_COMMIT=d60cd09
ARG BASIX_GIT_COMMIT=a702f51
ARG FFCX_GIT_COMMIT=bd29ed3
ARG DOLFINX_GIT_COMMIT=146860e

ENV PYTHONDONTWRITEBYTECODE=1

RUN pip3 install git+https://github.com/FEniCS/ufl.git@$UFL_GIT_COMMIT \
    && \
    pip3 install git+https://github.com/FEniCS/basix.git@$BASIX_GIT_COMMIT \
    && \
    pip3 install git+https://github.com/FEniCS/ffcx.git@$FFCX_GIT_COMMIT

RUN git clone --branch main https://github.com/FEniCS/dolfinx.git \
    && \
    cd dolfinx \
    && \
    git checkout $DOLFINX_GIT_COMMIT \
    && \
    mkdir -p build && cd build \
    && \
    PETSC_ARCH=linux-gnu-real-32 cmake -DCMAKE_BUILD_TYPE=$DOLFINY_BUILD_TYPE ../cpp/ \
    && \
    PETSC_ARCH=linux-gnu-real-32 make -j`nproc` install \
    && \
    make clean \
    && \
    cd ../python \
    && \
    PETSC_ARCH=linux-gnu-real-32 pip3 -v install . --user

RUN pip3 install matplotlib
