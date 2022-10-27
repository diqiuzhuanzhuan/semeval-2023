FROM nvidia/cuda:11.8.0-base-ubuntu22.04
RUN apt update 
RUN pip config set global.index-url 'https://pypi.tuna.tsinghua.edu.cn/simple'
ADD . requirements.txt
RUN pip install -r requirements.txt