




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
This section describes the command line flags associated with each step in the workflow; for a complete, worked example with code, see :doc:`vignette`.

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

Although ``disperseNN2`` does not actually run simulations, it relies on simulated training data. Therefore, we provide some template code for generating training data; however, the ideal analysis will tailor the simulation step to take advantage of realistic information from your particular study system. For information on how to implement population genetic simulations, check out the extensive `SLiM manual <http://benhaller.com/slim/SLiM_Manual.pdf>`_.

The simulation script we used for validating ``disperseNN2`` is ``SLiM_recipes/bat20.slim``. This is a continuous space model where the mother-offspring distance is :math:`N(0,\sigma)` in both the x and y dimensions, and is described in detail in `Battey et al. 2020 <https://doi.org/10.1534/genetics.120.303143>`_. Below is an example simulation command:

.. code-block:: bash

		mkdir -p temp_wd/TreeSeqs
		
		slim -d SEED=12345 \
		     -d sigma=0.2 \
		     -d K=5 \
		     -d mu=0 \
		     -d r=1e-8 \
		     -d W=50 \
		     -d G=1e8 \
		     -d maxgens=100000 \
		     -d OUTNAME="'temp_wd/TreeSeqs/my_sequence'" \
		     SLiM_recipes/bat20.slim \
		     # Note the two sets of quotes around the output name
		
Command line arguments are passed to ``SLiM`` using the ``-d`` flag followed by the variable name as it appears in the recipe file.

- ``SEED`` - a random seed to reproduce the simulation results.
- ``sigma`` - the dispersal parameter.
- ``K`` - carrying capacity. Note: the carrying capacity in this model, K, corresponds roughly to density, but the actual density will vary depending on the model,and will fluctuate a bit over time.
- ``mu`` - per base per genertation mutation rate.
- ``r`` -  per base per genertation recombination rate.
- ``W`` - the height and width of the geographic spatial boundaries.
- ``G`` - total size of the simulated genome.
- ``maxgens`` - number of generations to run simulation.
- ``OUTNAME`` - prefix to name output files.

After running ``SLiM`` for a fixed number of generations, the simulation is still not complete, as many trees will likely not have coalesced still. Next you will need to finish, or "recapitate", the tree sequences. We recommend recapitating up front, as training is prohibitively slow if you try to recapitate on-the-fly. The below code snippet in python can be used to recapitate a tree sequence:

.. code-block:: python

		import tskit,msprime
		ts=tskit.load("temp_wd/TreeSeqs/my_sequence_12345.trees")
		Ne=len(ts.individuals())
		demography = msprime.Demography.from_tree_sequence(ts)
		demography[1].initial_size = Ne
		ts = msprime.sim_ancestry(initial_state=ts, recombination_rate=1e-8, demography=demography, start_time=ts.metadata["SLiM"]["cycle"],random_seed=12345)
		ts.dump("temp_wd/TreeSeqs/my_sequence_12345_recap.trees")


For planning the total number of simulations, consider the following things. First: you can get away with fewer simulations by taking repeated, pseudo-independent samples from each simulation—--that is, if the simulated populations are sufficiently large relative to the sample size. Second: if the simulatios explore a large parameter space, e.g. more than	one or two free	parameters, then largertraining sets may be required.	In our analysis, we ran 1000 simulations while varying only the dispersal rate parameter, and sample 50	times from each	simulation (see Preprocessing, below).

The only real requirements of ``disperseNN2`` regarding training data are: genotypes are in a 2D array, the corresponding sample locations are in a table with two columns, and the targets are in a table with one column; all as numpy arrays. Therefore, simulation programs other than ``SLiM`` could be used in theory. However, given the strict format of the input files, we do not recommend users attempt to generate training data from sources other than ``SLiM``. 









.. _preprocessing:

****************
2. Preprocessing
****************

The preprocessing step involves more simulation, technically: it adds mutations to each tree sequence, takes a sample of individuals, and saves the genotypes and sample locations in numpy arrays.
This speeds up training.
In addition, multiple samples can be taken from the same tree sequence to make the training set larger.
A basic preprocessing command looks like:

.. code-block:: bash
		
		python disperseNN2.py \
                       --out temp_wd/output_dir \
		       --preprocess \
                       --num_samples 10 \
		       --num_snps 5000 \
		       --n 10 \
		       --seed 1 \
		       --edge_width 3 \
		       --tree_list Examples/tree_list1.txt \
		       --target_list Examples/target_list1.txt

- ``--out``: output directory
- ``--preprocess``: this flag tells ``disperseNN2`` to preprocess the training data
- ``--num_samples``: this is the number of independent samples to take from each tree sequence; the total training set will be the number of simulated tree sequences :math:`\times` the number of samples from each.
- ``--num_snps``: the number of SNPs to use as input for the CNN
- ``--n``: sample size
- ``--seed``: random number seed
- ``--edge_width``: width of habitat edge to avoid sampling from
- ``--tree_list Examples/tree_list1.txt``: list of filepaths to the tree sequences
- ``--target_list Examples/target_list1.txt``: list of filepaths to .txt files with the target values
  
The preprocessing step can be parallelized to some extent: a single command preprocesses all simulations serially by taking one sample of genotypes from each dataset, so independent commands can be used with different random number seeds to take multiple, pseudo-independent samples from each simulation.
		
The preprocessed data are saved in the directory specified by ``--out``; Subsequent outputs will also be saved in this folder.







.. _training:

***********
3. Training
***********

Below is an example command for the training step.
This example uses tree sequences as input.

.. code-block:: bash

		python disperseNN2.py \
		       --out temp_wd/output_dir \
		       --train \
		       --preprocessed \
		       --num_snps 5000 \
		       --max_epochs 10 \
		       --validation_split 0.2 \
		       --batch_size 10 \
		       --threads 1 \
		       --seed 12345 \
		       --n 10 \
		       --learning_rate 1e-4 \
		       --pairs 45 \
		       --pairs_encode 45 \
		       --pairs_estimate 45 \
		       > temp_wd/output_dir/training_history.txt

- ``--train``: tells ``disperseNN2`` to train a neural network
- ``--preprocessed``: tells ``disperseNN2`` to use already-preprocessed data, which it looks for in the ``--out`` directory.
- ``--max_epochs``: maximum number of epochs to train for.
- ``--validation_split``: the proportion of training data held out for validation between batches for hyperparameter tuning.
- ``--batch_size``: we find that batch_size=10 works well.
- ``--threads``: number of threads to use with the multiprocessor. 
- ``--learning_rate``: learning rate to use during training. It's scheduled to decrease by 2x every 10 epochs with no decrease in validation loss.
- ``--pairs``: the total number of pairs to include in the analysis
- ``--pairs_encode``: the number of pairs to include in the gradient in the encoder portion of the neural network.
- ``--pairs_estimate``: the number of pairs to include in the estimator portion of the neural network.

This command will print the training progress to stdout, which was redirected to ``temp_wd/output_dir/training_history.txt`` in this example.
The model weights are saved to ``temp_wd/output_dir/out_12345_model.hdf5``.
In practice, you will need a training set of maybe 50,000, and you will likely want to train for longer than 10 epochs.
For reading preprocessed training data we recommend trying between 1 and 10 threads. 







.. _validation:

*************
4. Validation
*************

If you want to predict :math:`\sigma` from simulated tree sequences output by ``SLiM``, a predict command like the below one can be used:

.. code-block:: bash

		python disperseNN2.py \
		       --out temp_wd/output_dir \
		       --predict \
		       --preprocessed \
		       --num_snps 5000 \
		       --batch_size 1 \
		       --threads 1 \
		       --n 10 \
		       --seed 12345 \
		       --pairs 45 \
		       --pairs_encode 45 \
		       --pairs_estimate 45 \
		       --load_weights temp_wd/output_dir/out_12345_model.hdf5 \
		       --num_pred 5

- ``--predict``: tells ``disperseNN2`` to perform predictions
- ``--load_weights``: loads in saved weights from an already-trained model
- ``--num_pred``: number of datasets to predict with.

Similar to the earlier prediction example, this will generate a file called ``temp_wd/output_dir/Test_12345/out_12345_predictions.txt`` containing (TO DO: random number seeds aren't reproducible):

.. code-block:: bash

		0.1690090249743872      0.48620286613483377
		0.6280568409720466      0.4672472252013161
		0.7184737596020008      0.13608900222161735
		-0.7790530578965832     0.23677401340070897
		-0.27202587929510147    -0.01729259869841701

Here, the second and third columns contain the true and predicted :math:`\sigma`; for each simulation.









.. _empirical:

************************
5. Empirical predictions
************************

For predicting with empirical data, the command will be slightly different: instead of a list of tree sequences (and targets?), a new flag is given, --empirical, which is a prefix for two files: a VCF and a table of lat and long. The lat and longs get projected onto a flat 2D map using ____. (TODO: empirical estimation)


.. code-block:: bash

                python disperseNN2.py \
		       --out temp_wd/output_dir \
		       --predict \
		       --empirical Examples/VCFs/halibut \
		       --num_snps 5000 \
		       --batch_size 1 \
		       --threads 1 \
		       --n 10 \
		       --seed 12345 \
		       --pairs 45 \
		       --pairs_encode 45 \
		       --pairs_estimate 45 \
		       --load_weights temp_wd/output_dir/out_12345_model.hdf5 \
		       --num_pred 1

		
