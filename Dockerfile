FROM dolfinx/dev-env:current

ARG DOLFINY_BUILD_TYPE="RelWithDebInfo"

ARG UFL_GIT_COMMIT="0f2cb21"
ARG BASIX_GIT_COMMIT="2e2a704"
ARG FFCX_GIT_COMMIT="9b0320a"
ARG DOLFINX_GIT_COMMIT="c7c6205"

ARG PYPI_PACKAGE_REPO="https://gitlab.uni.lu/api/v4/projects/3415/packages/pypi/simple"

ARG CPPYY_VERSION="3.1.2"
ARG PYVISTA_VERSION="0.43.3"

ARG PETSC_ARCH="linux-gnu-real64-32"

RUN git clone --branch main https://github.com/FEniCS/basix.git /fenics/basix \
        && cd /fenics/basix \
        && git checkout ${BASIX_GIT_COMMIT} \
        && cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFINY_BUILD_TYPE} -B build -S ./cpp \
        && cmake --build build \
        && cmake --install build \
        && pip3 install \
            --config-settings=cmake.build-type=${DOLFINY_BUILD_TYPE} --config-settings=install.strip=false \
            --no-cache-dir --no-build-isolation --no-dependencies \
            ./python \
    && \
    git clone --branch main https://github.com/FEniCS/ufl.git /fenics/ufl \
        && cd /fenics/ufl \
        && git checkout ${UFL_GIT_COMMIT} \
        && pip3 install \
            --no-cache-dir --no-build-isolation --no-dependencies \
            . \
    && \
    git clone --branch main https://github.com/FEniCS/ffcx.git /fenics/ffcx \
        && cd /fenics/ffcx \
        && git checkout ${FFCX_GIT_COMMIT} \
        && pip3 install \
            --no-cache-dir --no-build-isolation --no-dependencies \
            . \
    && \
    git clone --branch main https://github.com/FEniCS/dolfinx.git /fenics/dolfinx \
        && cd /fenics/dolfinx \
        && git checkout ${DOLFINX_GIT_COMMIT} \
        && cmake -G Ninja -DCMAKE_BUILD_TYPE=${DOLFINY_BUILD_TYPE} -B build -S ./cpp \
        && cmake --build build \
        && cmake --install build \
        && pip3 install \
            --config-settings=cmake.build-type=${DOLFINY_BUILD_TYPE} --config-settings=install.strip=false \
            --no-cache-dir --no-build-isolation --no-dependencies \
            ./python \
    && \
    pip3 install --no-cache-dir --index-url ${PYPI_PACKAGE_REPO} \
        cppyy==${CPPYY_VERSION} \
        pyvista==${PYVISTA_VERSION} \
    && \
    pip3 cache purge \
    && \
    find / -type f -name 'allDict.cxx*' -delete && EXTRA_CLING_ARGS="-Ofast" python3 -c 'import cppyy'  # precompiled headers

ENV PYTHONDONTWRITEBYTECODE=1
ENV PETSC_ARCH=${PETSC_ARCH}
ENV EXTRA_CLING_ARGS="-Ofast"
