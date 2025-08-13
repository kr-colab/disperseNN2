

.. _gpus:

GPUs
----

Since tensorflow and cuda versions must be compatible with the particular `NVIDIA drivers <https://www.tensorflow.org/install/source#gpu>`_ on your computer, there may be no single set of instructions that will work for everyone. 

The below command has worked for us (after following the basic disperseNN2 install istructions):

.. code-block:: console

                (.venv) $ pip install tensorflow[and-cuda]

To test that your installation works, train with ``disperseNN2`` (see  :doc:`vignette`.), and in a separate window run `gpustat` to make sure the GPU is actually firing. It isn't sufficient to just "pick up" the GPU, because often bits of cuda code are still missing and the GPU remains unused.

