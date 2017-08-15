# Authors:
# Padmec

FROM padmec/elliptic:1.0
MAINTAINER Maryna Diógenes <marynadiogenes@gmail.com>

RUN easy_install pytest configobj
RUN git clone https://github.com/padmec-reservoir/PRESTO.git
WORKDIR $HOME/PRESTO

RUN python setup.py build
RUN python setup.py install
