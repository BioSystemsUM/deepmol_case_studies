FROM nvcr.io/nvidia/tensorflow:23.11-tf2-py3

WORKDIR /workspace

COPY ./requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt

RUN echo "ola"
RUN pip install --no-deps git+https://github.com/BioSystemsUM/DeepMol.git@new_tensorflow_version#egg=deepmol

RUN pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec

COPY . /workspace

RUN pip install . --no-deps

WORKDIR /workspace/scripts

CMD bash