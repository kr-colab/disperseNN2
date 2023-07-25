

.. _gpus:

GPUs
----

Since tensorflow and cuda versions must be compatible with the particular `NVIDIA drivers <https://www.tensorflow.org/install/source#gpu>`_ on your computer, there is no single set of instructions that will work for everyone. We use the below commands to set things up on our computer. These will hopefully work for you, or at least serve as a startng point, but you may need to improvise.

Some commands must be run individually, so don't copy the whole code block:

.. code-block:: console

                (.venv) $ mamba install cudatoolkit=11.8.0 cuda-nvcc -c conda-forge -c nvidia --yes
                (.venv) $ python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
                (.venv) $ mkdir -p $CONDA_PREFIX/bin/nvvm/libdevice/
                (.venv) $ ln -s $CONDA_PREFIX/nvvm/libdevice/libdevice.10.bc $CONDA_PREFIX/bin/nvvm/libdevice/
                (.venv) $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
                (.venv) $ echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
                (.venv) $ echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
                (.venv) $ echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/bin/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
                (.venv) $ source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
                (.venv) $ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" # verify that gpus get picked up
