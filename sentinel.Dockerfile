FROM ubuntu:20.04

RUN apt-get update && apt-get install -y wget git gcc

# place to keep our app and the data
RUN mkdir -p /app /app/tails /app/models /data /data/logs /_tmp

# install miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O miniconda.sh && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    apt-get clean

# copy over the config and the code
COPY ["config.yaml", "setup.py", "ztf-sentinel-service/", "ztf-sentinel-service/tests/", "/app/"]
COPY ["tails", "/app/tails"]
COPY ["models", "/app/models"]

WORKDIR /app

# install service requirements, swarp, and tails; generate supervisord conf file
RUN /opt/conda/bin/conda install -c conda-forge astromatic-swarp && \
    ln -s /opt/conda/bin/swarp /bin/swarp && \
    /opt/conda/bin/pip install -U pip cython && \
    /opt/conda/bin/python setup.py install && \
    /opt/conda/bin/pip install -r requirements.txt --no-cache-dir && \
    /opt/conda/bin/python generate_supervisor_conf.py

# run container
CMD /opt/conda/bin/supervisord -n -c supervisord_watcher.conf
