FROM python:latest
MAINTAINER Arthur Mensch <arthur.mensch@gmail.com

# Configure environment
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir -p /data/nilearn_data && mkdir /cache \
 && mkdir -p /usr/src/modl && mkdir -p /root/examples
VOLUME /data
VOLUME /cache
VOLUME /root/examples

ENV NILEARN_DATA=/data/nilearn_data
ENV DATA=/data
ENV CACHE=/cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
&& rm -rf ~/.cache/pip/ && rm requirements.txt

# Configure matplotlib to avoid using QT
COPY misc/matplotlibrc /root/.config/matplotlib/matplotlibrc
# Trigger creation of the matplotlib font cache
ENV MATPLOTLIBRC=/work/.config/matplotlib
RUN python -c "import matplotlib.pyplot"

COPY . /usr/src/modl/
WORKDIR /usr/src/modl
RUN python setup.py install

WORKDIR /root
ENTRYPOINT ["/usr/local/bin/python"]
