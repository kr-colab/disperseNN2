

.. _gpus:

GPUs
----

Since tensorflow and cuda versions must be compatible with the particular `NVIDIA drivers <https://www.tensorflow.org/install/source#gpu>`_ on your computer, there is no single set of instructions that will work for everyone. In fact, since the `disperseNN2` code requires a newer tensorflow, our code is simply not compatible with older GPU setups.

We use the below commands to set things up on our computer. These will hopefully work for you, or at least serve as a startng point, but you may need to improvise.

.. code-block:: console

                (.venv) $ conda install cudatoolkit=11.8.0 cuda-nvcc -c conda-forge -c nvidia --yes # or use mamba
                (.venv) $ python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
                (.venv) $ mkdir -p $CONDA_PREFIX/bin/nvvm/libdevice/
                (.venv) $ ln -s $CONDA_PREFIX/nvvm/libdevice/libdevice.10.bc $CONDA_PREFIX/bin/nvvm/libdevice/
                (.venv) $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
                (.venv) $ echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
                (.venv) $ echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
                (.venv) $ echo 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/bin/' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
                (.venv) $ source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

To test that your installation works, train with `disperseNN2` (see  :doc:`vignette`.), and in a separate window run `gpustat` to make sure the GPU is actually firing. It isn't sufficient to just "pick up" the GPU, because often bits of cuda code are still missing and the GPU remains unused.

