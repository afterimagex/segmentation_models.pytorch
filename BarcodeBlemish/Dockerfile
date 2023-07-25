# ===== source =====

FROM jupyter/base-notebook:latest
USER root

RUN codename=$(. /etc/os-release;echo $VERSION_CODENAME) && \
    echo -e "deb http://mirrors.aliyun.com/ubuntu/ ${codename} main restricted universe multiverse\n \
    deb-src http://mirrors.aliyun.com/ubuntu/ ${codename} main restricted universe multiverse\n \
    deb http://mirrors.aliyun.com/ubuntu/ ${codename}-updates main restricted universe multiverse\n \
    deb-src http://mirrors.aliyun.com/ubuntu/ ${codename}-updates main restricted universe multiverse\n \
    deb http://mirrors.aliyun.com/ubuntu/ ${codename}-backports main restricted universe multiverse\n \
    deb-src http://mirrors.aliyun.com/ubuntu/ ${codename}-backports main restricted universe multiverse\n \
    deb http://mirrors.aliyun.com/ubuntu/ ${codename}-proposed main restricted universe multiverse\n \
    deb-src http://mirrors.aliyun.com/ubuntu/ ${codename}-proposed main restricted universe multiverse\n \
    deb http://mirrors.aliyun.com/ubuntu/ ${codename}-security main restricted universe multiverse\n \
    deb-src http://mirrors.aliyun.com/ubuntu/ ${codename}-security main restricted universe multiverse\n" > /etc/apt/sources.list

RUN conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free && \
    conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main && \
    conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/conda-forge && \
    conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/Paddle && \
    conda config --set show_channel_urls yes

RUN pip install --no-cache-dir -U pip && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# ===== install =====

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-dev \
    wget \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/.requirements.txt

RUN pip install --no-cache-dir -r /tmp/.requirements.txt

WORKDIR /workspace

USER ${NB_UID}