FROM nvcr.io/nvidia/tensorflow:23.11-tf2-py3

WORKDIR /workspace

COPY ./requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt

RUN pip install --no-deps deepmol[all]==1.1.1

RUN pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec

COPY . /workspace

RUN pip install . --no-deps

WORKDIR /workspace/scripts

CMD bash