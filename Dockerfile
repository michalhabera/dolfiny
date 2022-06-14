FROM dolfinx/dev-env

ARG DOLFINY_BUILD_TYPE=Release

ARG UFL_GIT_COMMIT=2ad4ef16f9149f0c079b0d1fd7c63c07ce3c8700
ARG BASIX_GIT_COMMIT=314498fc9a3a3cc398baa8dc921212d96f647a0e
ARG FFCX_GIT_COMMIT=61a02c26c97a096f24857feca008c7c49b7616ea
ARG DOLFINX_GIT_COMMIT=cefa0092702e80e254c7a8e58d6af17f05b12a8d

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

RUN pip3 install matplotlib
