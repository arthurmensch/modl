FROM python:latest
MAINTAINER Arthur Mensch <arthur.mensch@gmail.com

# Configure environment
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install --no-cache-dir ipython

RUN mkdir -p /usr/src/modl
WORKDIR /usr/src

COPY requirements.txt modl
RUN pip install --no-cache-dir -r modl/requirements.txt

# Configure matplotlib to avoid using QT
COPY matplotlibrc /work/.config/matplotlib/matplotlibrc

# Trigger creation of the matplotlib font cache
ENV MATPLOTLIBRC=/work/.config/matplotlib
RUN python -c "import matplotlib.pyplot"

COPY . modl
WORKDIR /usr/src/modl
RUN python setup.py install
WORKDIR /usr/src/modl/examples

ENTRYPOINT "/usr/local/bin/ipython"
