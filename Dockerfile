FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04
ARG access_key_id
ARG secret_access_key

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils
RUN apt-get install --no-install-recommends ffmpeg libsm6 libxext6 -y
WORKDIR /app

COPY . /app

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install "dvc[s3]"
RUN dvc config remote.ycloud.access_key_id $access_key_id && \
    dvc config remote.ycloud.secret_access_key $secret_access_key && \
    dvc pull && \
    dvc config remote.ycloud.access_key_id -u && \
    dvc config remote.ycloud.secret_access_key -u && \
    rm -rf *.dvc

RUN python3.8 -m pip install -r requirements.txt
RUN python3.8 -m pip install packages/torch-1.7.1+cu110-cp38-cp38-linux_x86_64.whl packages/torchvision-0.8.2+cu110-cp38-cp38-linux_x86_64.whl

CMD ["uvicorn", "yolo:app", "--app-dir=./", "--reload", "--workers=1", "--host=0.0.0.0", "--port", "1000", "--use-colors", "--loop=uvloop"]










