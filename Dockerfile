FROM nvcr.io/nvidia/tensorflow:23.07-tf2-py3

# RUN wget https://repo.continuum.io/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -O miniconda.sh && \
#     chmod a+x miniconda.sh && \
#     ./miniconda.sh -b -p /opt/conda && \
#     rm miniconda.sh

# COPY ./dgl /workspace/dgl
# WORKDIR /workspace/dgl
# # Configure environment
# ENV CONDA_DIR=/opt/conda \
#     SHELL=/bin/bash \
#     LC_ALL=en_US.UTF-8 \
#     LANG=en_US.UTF-8 \
#     LANGUAGE=en_US.UTF-8

# ENV PATH=$CONDA_DIR/bin:$PATH \
#     HOME=/home

# # RUN bash /workspace/dgl/script/create_dev_conda_env.sh -c -s
# RUN conda init bash \
#     && . ~/.bashrc \
#     && bash /workspace/dgl/script/create_dev_conda_env.sh -g 12.2 -t 2.0.1 -s \
#     && conda activate dgl-dev-gpu-122 
# # RUN bash /workspace/dgl/script/create_dev_conda_env.sh -h -s

# # RUN echo "source activate dgl-dev-gpu-122" > ~/.bashrc
# # ENV PATH /opt/conda/envs/dgl-dev-gpu-122/bin:$PATH
# # RUN ~/.bashrc

# ENV DGL_HOME=/workspace/dgl
# # RUN bash /workspace/dgl/script/build_dgl.sh -c
# RUN bash /workspace/dgl/script/build_dgl.sh -g
# # RUN bash /workspace/dgl/script/build_dgl.sh -h
# RUN python /workspace/dgl/python/setup.py install
# # Build Cython extension
# RUN python /workspace/dgl/python/setup.py build_ext --inplace

WORKDIR /workspace

COPY ./requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt

COPY ./deepmol-1.0.7.1-py3-none-any.whl /workspace/deepmol-1.0.7.1-py3-none-any.whl
RUN pip install --no-deps deepmol-1.0.7.1-py3-none-any.whl
RUN pip install git+https://github.com/samoturk/mol2vec#egg=mol2vec

COPY . /workspace

RUN pip install . --no-deps

WORKDIR /workspace/scripts

CMD ["python", "HIA_Hou/hia_hou.py"]