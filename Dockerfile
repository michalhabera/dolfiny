FROM dolfinx/dev-env:current

ARG DOLFINY_BUILD_TYPE="RelWithDebInfo"

ARG UFL_GIT_COMMIT="cc6c5a87e1fed1ea2357e0e84a2f46a4e72d684d"
ARG BASIX_GIT_COMMIT="0fc130f7d790f793ffa57bf8d056232ff6f12be8"
ARG FFCX_GIT_COMMIT="2fb6a24c6a85f842fd26a06520dbccfeac120f8c"
ARG DOLFINX_GIT_COMMIT="56fad9f56a3f6c0537b9fde7f48b1d20be2f101f"

ARG PYPI_PACKAGE_REPO="https://gitlab.uni.lu/api/v4/projects/3415/packages/pypi/simple"

ENV PYTHONDONTWRITEBYTECODE=1
ENV CLING_REBUILD_PCH=1
ENV PETSC_ARCH="linux-gnu-real64-32"

RUN git clone --branch main https://github.com/FEniCS/basix.git \
        && cd basix \
        && git checkout ${BASIX_GIT_COMMIT} \
        && cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFINY_BUILD_TYPE} -B build -S ./cpp \
        && cmake --build build \
        && cmake --install build \
        && python3 -m pip install ./python \
    && \
    pip3 install git+https://github.com/FEniCS/ufl.git@${UFL_GIT_COMMIT} \
    && \
    pip3 install git+https://github.com/FEniCS/ffcx.git@${FFCX_GIT_COMMIT}

RUN git clone --branch main https://github.com/FEniCS/dolfinx.git \
        && cd dolfinx \
        && git checkout ${DOLFINX_GIT_COMMIT} \
        && cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFINY_BUILD_TYPE} -B build -S ./cpp \
        && cmake --build build \
        && cmake --install build \
        && python3 -m pip install ./python

RUN pip3 install --index-url ${PYPI_PACKAGE_REPO} cppyy==3.0.0
