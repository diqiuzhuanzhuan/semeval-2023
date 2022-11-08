FROM pytorchlightning/pytorch_lightning:latest
RUN pip config set global.index-url 'https://pypi.tuna.tsinghua.edu.cn/simple'
ADD requirements.txt .
RUN pip install --upgrade pip &&  pip install -r requirements.txt