FROM dolfinx/dev-env:v0.5.1

ARG DOLFINY_BUILD_TYPE=Release

ARG UFL_GIT_COMMIT=50d2a4863ed6bc46e35c03a97193ee97fd09ceaf
ARG BASIX_GIT_COMMIT=8f731b9f76b9d72d25f799dfc7b40cb21f50641e
ARG FFCX_GIT_COMMIT=b057deadac4fbe30db6472e92eca5f91b2d0cb4d
ARG DOLFINX_GIT_COMMIT=136aa0acdeeb7e29b336949ad9ed8537237f5f88

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
