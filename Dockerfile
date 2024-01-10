FROM dolfinx/dev-env:current

ARG DOLFINY_BUILD_TYPE="RelWithDebInfo"

ARG UFL_GIT_COMMIT="7277fd1"
ARG BASIX_GIT_COMMIT="b935d2d"
ARG FFCX_GIT_COMMIT="2d60a8b"
ARG DOLFINX_GIT_COMMIT="75ba784"

ARG PYPI_PACKAGE_REPO="https://gitlab.uni.lu/api/v4/projects/3415/packages/pypi/simple"

ENV PYTHONDONTWRITEBYTECODE=1
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

RUN pip3 install --index-url ${PYPI_PACKAGE_REPO} cppyy==3.1.2
