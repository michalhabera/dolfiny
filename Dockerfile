FROM dolfinx/dev-env

ARG DOLFINY_BUILD_TYPE=Release

ENV PETSC_ARCH=linux-gnu-real-32 

RUN pip3 install git+https://github.com/FEniCS/ufl.git --upgrade \
    && \
    pip3 install git+https://github.com/FEniCS/basix.git --upgrade \
    && \
    pip3 install git+https://github.com/FEniCS/ffcx.git --upgrade \
    && \
    rm -rf /usr/local/include/dolfin /usr/local/include/dolfin.h

RUN git clone --branch master https://github.com/FEniCS/dolfinx.git \
    && \
    cd dolfinx \
    && \
    mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=$DOLFINY_BUILD_TYPE ../cpp/ \
    && \
    make -j`nproc` install \
    && \
    cd /

RUN cd dolfinx/python \
    && \
    pip3 -v install . --user

ENV PYTHONDONTWRITEBYTECODE 1
