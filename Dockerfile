FROM dolfinx/dev-env:nightly

ARG DOLFINY_BUILD_TYPE=Release

ARG UFL_GIT_COMMIT=50d2a48
ARG BASIX_GIT_COMMIT=8f731b9
ARG FFCX_GIT_COMMIT=763c6be
ARG DOLFINX_GIT_COMMIT=2b366d

ENV PYTHONDONTWRITEBYTECODE=1

RUN git clone --branch main https://github.com/FEniCS/basix.git \
        && cd basix \
        && git checkout $BASIX_GIT_COMMIT \
        && cmake -G Ninja -DCMAKE_BUILD_TYPE=$DOLFINY_BUILD_TYPE -B build -S ./cpp \
        && cmake --build build \
        && cmake --install build \
        && python3 -m pip install ./python \
    && \
    pip3 install git+https://github.com/FEniCS/ufl.git@$UFL_GIT_COMMIT \
    && \
    pip3 install git+https://github.com/FEniCS/ffcx.git@$FFCX_GIT_COMMIT

RUN git clone --branch main https://github.com/FEniCS/dolfinx.git \
        && cd dolfinx \
        && git checkout $DOLFINX_GIT_COMMIT \
        && export PETSC_ARCH=linux-gnu-real-32 \
        && cmake -G Ninja -DCMAKE_BUILD_TYPE=$DOLFINY_BUILD_TYPE -B build -S ./cpp \
        && cmake --build build \
        && cmake --install build \
        && python3 -m pip install ./python

RUN pip3 install matplotlib cppyy
