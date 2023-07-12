




.. _usage:

Usage
-----



.. _install:

Install
^^^^^^^

To use ``disperseNN2``, first install it using pip:

.. code-block:: console

   (.venv) $ pip install ``disperseNN2``



Workflow
^^^^^^^^
This section describes the command line flags associated with each step in the workflow; for a complete, worked example, see :doc:`vignette`.

A typical ``disperseNN2`` workflow involves five steps:

.. While it might be possible to run smaller tests on a laptop, it is generally advisable to seek out a high performance computing cluster, particularly for the simulation step.                                                                                                                                                     

:ref:`simulation`
   
:ref:`preprocessing`

:ref:`training`

:ref:`validation`

:ref:`empirical`



     



.. _simulation:

*************   
1. Simulation
*************

Although ``disperseNN2`` is not used for running simulations, it relies on simulated training data. Therefore, we provide some template code for generating training data. Hhowever, the ideal analysis will tailor the simulation step to take advantage of realistic information from your particular study system. For information on how to implement population genetic simulations, check out the `SLiM manual <http://benhaller.com/slim/SLiM_Manual.pdf>`_.

The simulation script we use to train ``disperseNN2`` is ``SLiM_recipes/square.slim``. This is a continuous space model where the mother-offspring distance is :math:`N(0,\sigma)` in both the :math:`x` and :math:`y` dimensions. Other details of the model are described in `Battey et al. 2020 <https://doi.org/10.1534/genetics.120.303143>`_. Below is an example simulation command:

.. code-block:: bash

		mkdir -p temp_wd/TreeSeqs
		
		slim -d SEED=12345 \
		     -d sigma=0.2 \
		     -d K=10 \
		     -d r=1e-8 \
		     -d W=50 \
		     -d G=1e8 \
		     -d maxgens=1000 \
		     -d OUTNAME="'temp_wd/TreeSeqs/my_sequence'" \
		     SLiM_recipes/square.slim \
		     # Note the two sets of quotes around the output name
		
Command line arguments are passed to ``SLiM`` using the ``-d`` flag followed by the variable name as it appears in the recipe file.

- ``SEED``: a random seed to reproduce the simulation results.
- ``sigma``: the dispersal parameter.
- ``K``: carrying capacity. Note: the carrying capacity in this model, K, corresponds roughly to density, but the actual density will vary depending on the model, and will fluctuate a bit over time.
- ``r``:  per base per genertation recombination rate.
- ``W``: the height and width of the geographic spatial boundaries.
- ``G``: total size of the simulated genome.
- ``maxgens``: number of generations to run simulation.
- ``OUTNAME``: prefix to name output files.

.. note::

   The above example used only 1,000 spatial generations; this strategy should be used with caution because this can affect how the output is interpreted. In addition, isolation-by-distance is usually weaker with fewer spatial generations which reduces signal for dispersal rate. In the ``disperseNN2`` paper we ran 100,000 generations spatial.

  
After running ``SLiM`` for a fixed number of generations, the simulation is still not complete, as many trees will likely not have coalesced still. Next you will need to finish, or "recapitate", the tree sequences. We recommend recapitating at this early stage, before training, as training can be prohibitively slow if you recapitate on-the-fly. The below code snippet in python can be used to recapitate a tree sequence:

.. code-block:: python

		import tskit,msprime
		ts=tskit.load("temp_wd/TreeSeqs/my_sequence_12345.trees")
		Ne=len(ts.individuals())
		demography = msprime.Demography.from_tree_sequence(ts)
		demography[1].initial_size = Ne
		ts = msprime.sim_ancestry(initial_state=ts, recombination_rate=1e-8, demography=demography, start_time=ts.metadata["SLiM"]["cycle"],random_seed=12345)
		ts.dump("temp_wd/TreeSeqs/my_sequence_12345_recap.trees")

.. note::

   Here, we have assumed a constant demographic history. If an independently inferred demographic history for your species is available, or if you want to explore different demographic histories, the recapitation step is a good place for implementing these changes. For more information see the `msprime docs <https://tskit.dev/msprime/docs/stable/ancestry.html#demography>`_.


For planning the total number of simulations, consider the following. First, you might be able to get away with fewer simulations by taking repeated, pseudo-independent samples from each simulation. Second, if the simulations explore a large parameter space, e.g. more than	one or two free	parameters, then larger training sets may be required.	In our paper, we ran 1000 simulations while varying only the dispersal rate parameter, and sampled 50 times from each	simulation (see Preprocessing, below).

Simulation programs other than ``SLiM`` could be used in theory. The only real requirements of ``disperseNN2`` regarding training data are: genotypes are in a 2D array, the corresponding sample locations are in a table with two columns, and the target values are saved in individual files; all as numpy arrays. 









.. _preprocessing:

****************
2. Preprocessing
****************

The preprocessing step actually involves more simulation: it adds mutations to each tree sequence, takes a sample of individuals, and then saves the genotypes and sample locations in numpy arrays.
Doing these steps up front instaed of during training is more efficient.
In addition, multiple samples can be taken from the same tree sequence to make the training set larger.
A basic preprocessing command looks like:

.. code-block:: bash
		
		python disperseNN2.py \
                       --out temp_wd/output_dir \
		       --preprocess \
                       --n 10 \
		       --num_snps 5000 \
		       --tree_list Examples/tree_list1.txt \
		       --target_list Examples/target_list1.txt \
		       --empirical Examples/VCFs/halibut \
		       --seed 1
		       

- ``--out``: output directory
- ``--preprocess``: this flag tells ``disperseNN2`` to preprocess the training data
- ``--n``: sample size
- ``--num_snps``: the number of SNPs to use as input for the CNN
- ``--tree_list``: path to a list of filepaths to the tree sequences
- ``--target_list``: path to list of filepaths to .txt files with the target values
- ``--empirical``: prefix for the empirical locations. This includes the path, but without the filetype suffix, ".locs".
- ``--seed``: random number seed

.. note::

   Simulated individuals are sampled near the empirical sample locations. Our strategy involves first projecting the latitude and longitude coordinates for each location onto a 2D surface, in kilometers. By default, the projected locations are repositioned to new, random areas of the training map before sampling individuals from those locations; this is making the assumption that the true habitat range is unknown and we want our predictions to be invariant to the position of the sampling area within the greater species distribution.

.. Last, the spatial coordinates are rescaled to :math:`(0,1)`, preserving aspect ratio, before being shown to the neural network as input.
  
The preprocessing step can be parallelized to some extent: a single command preprocesses all simulations serially by taking one sample of genotypes from each dataset. Independent commands can be used with different random number seeds to take multiple, pseudo-independent samples from each simulation.
		
The preprocessed data are saved in the directory specified by ``--out``; Subsequent outputs will also be saved in this folder.







.. _training:

***********
3. Training
***********

Below is an example command for the training step.

.. code-block:: bash

		python disperseNN2.py \
		       --out Examples/Preprocessed \
		       --train \
		       --num_snps 1951 \
		       --max_epochs 50 \
		       --validation_split 0.2 \
		       --batch_size 10 \
		       --threads 1 \
		       --seed 12345 \
		       --n 10 \
		       --learning_rate 1e-4 \
		       --pairs 45 \
		       --pairs_encode 45 \
		       --pairs_estimate 45 \
		       --gpu -1

- ``--train``: tells ``disperseNN2`` to train a neural network
- ``--max_epochs``: maximum number of epochs to train for.
- ``--validation_split``: the proportion of training data held out for validation between batches for hyperparameter tuning. We use 0.2.
- ``--batch_size``: we find that batch_size=10 works well.
- ``--threads``: number of threads to use with the multiprocessor. 
- ``--learning_rate``: learning rate to use during training. It's scheduled to decrease by 2x every 10 epochs with no decrease in validation loss.
- ``--pairs``: the total number of pairs to include in the analysis
- ``--pairs_encode``: the number of pairs to include in the gradient in the encoder portion of the neural network.
- ``--pairs_estimate``: the number of pairs to include in the estimator portion of the neural network.
- ``--gpu``: as an integer, specifies the GPU index (e.g., 0, 1, etc). "any" means take any available gpu. -1 means no GPU.

This command will print the training progress to stdout.
The model weights are saved to ``<out>/out_12345_model.hdf5``.
In practice, you will need a training set of maybe 50,000, and you will likely want to train for longer than 10 epochs.
A single thread should be sufficient for reading preprocessed training data, but you might try up to 10 threads. 







.. _validation:

*************
4. Validation
*************

If you want to predict :math:`\sigma` from simulated data, a predict command like the below one can be used:

.. code-block:: bash

		python disperseNN2.py \
		       --out Examples/Preprocessed \
		       --predict \
		       --num_snps 1951 \
		       --batch_size 10 \
		       --threads 1 \
		       --n 10 \
		       --seed 12345 \
		       --pairs 45 \
		       --pairs_encode 45 \
		       --pairs_estimate 45 \
		       --load_weights Examples/Preprocessed/pretrained_model.hdf5 \
		       --num_pred 10

- ``--predict``: tells ``disperseNN2`` to perform predictions
- ``--load_weights``: loads in saved weights from an already-trained model
- ``--num_pred``: number of datasets to predict with.

This will generate a file called ``<out>/Test_<seed>/pwConv_<seed>_predictions.txt`` containing: (TO DO: random number seeds aren't reproducible):

.. code-block:: bash

		1.3140451217006555	1.6492410246659885
		0.7865230997761491	0.8685257153042614
		0.47119165922491946	0.7649536491646727
		0.767575286929296	1.1743635040408027
		3.2702120633354514	2.849078431752082
		1.6979504272705979	2.077949785913849
		5.088607364715273	3.4579915900251597
		0.4510196845483516	0.6887610946330337
		1.2068913180526788	2.5440247780616168
		2.335507009283163	3.0874019344367993

Here, the columns list the true and predicted :math:`\sigma` for each simulation.









.. _empirical:

************************
5. Empirical prediction
************************

Finally, for predicting with empirical data:

.. code-block:: bash

                python disperseNN2.py \
		       --out Examples/Preprocessed/ \
		       --predict \
		       --empirical Examples/VCFs/halibut \
		       --num_snps 1951 \
		       --threads 1 \
		       --n 10 \
		       --seed 12345 \
		       --pairs 45 \
		       --pairs_encode 45 \
		       --pairs_estimate 45 \
		       --load_weights Examples/Preprocessed/pretrained_model.hdf5 \
		       --num_reps 5

- ``--empirical``: prefix for the empirical data. This includes the path, but without the filetype suffix. Two files must be present: a VCF and a table of lat and long. 
- ``--num_reps``: specifies how many bootstrap replicates to perform. Each replicate takes a random draw of num_snps SNPs from the VCF.

The output is in kilometers can be found in ``<out>/empirical_<seed>_predictions.txt``:

.. code-block:: bash

		Examples/VCFs/halibut_0 0.2970724416
		Examples/VCFs/halibut_1 0.2578380969
		Examples/VCFs/halibut_2 0.3252334316
		Examples/VCFs/halibut_3 0.2698406216
		Examples/VCFs/halibut_4 0.2880779185
