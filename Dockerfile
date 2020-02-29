FROM quay.io/fenicsproject/dolfinx:dev-env-real

RUN pip3 install git+https://github.com/FEniCS/fiat.git --upgrade \
    && \
	 pip3 install git+https://github.com/FEniCS/ufl.git --upgrade \
    && \
	 pip3 install git+https://github.com/FEniCS/ffcx.git --upgrade \
    && \
	 rm -rf /usr/local/include/dolfin /usr/local/include/dolfin.h

RUN git clone --branch master https://github.com/FEniCS/dolfinx.git \
    && \
	 cd dolfinx \
    && \
	 mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Developer ../cpp/ \
    && \
	 make -j3 install \
	 && \
	 cd /

RUN cd dolfinx/python \
    && \
    pip3 -v install . --user

ENV PYTHONDONTWRITEBYTECODE 1