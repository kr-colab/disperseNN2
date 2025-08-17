

.. _gpus:

GPUs
----

Since tensorflow and cuda versions must be compatible with the particular `NVIDIA drivers <https://www.tensorflow.org/install/source#gpu>`_ on your computer, there is no single set of instructions that will work for everyone. In fact, since the `disperseNN2` code requires a newer tensorflow, our code is simply not compatible with older GPU setups.

We use the below commands to set things up on our computer. These will hopefully work for you, or at least serve as a startng point, but you may need to improvise.

.. code-block:: console

                (.venv) $ wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.12.0.46_cuda12-archive.tar.xz
                (.venv) $ tar -xf cudnn-linux-x86_64-9.12.0.46_cuda12-archive.tar.xz
                (.venv) $ cp cudnn-linux-x86_64-9.12.0.46_cuda12-archive/include/cudnn*.h $CONDA_PREFIX/include/
                (.venv) $ cp cudnn-linux-x86_64-9.12.0.46_cuda12-archive/lib/libcudnn* $CONDA_PREFIX/lib/
                (.venv) $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
                (.venv) $ echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
                (.venv) $ source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
                (.venv) $ mamba install -c nvidia cuda-toolkit=12.3 --yes
		
To test that your installation works, train with `disperseNN2` (see  :doc:`vignette`.), and in a separate window run `gpustat` to make sure the GPU is actually firing. It isn't sufficient to just "pick up" the GPU, because often bits of cuda code are still missing and the GPU remains unused.
