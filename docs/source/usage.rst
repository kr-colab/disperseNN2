




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
A typical ``disperseNN2`` workflow involves five steps:


:ref:`simulation`
   
:ref:`preprocessing`

:ref:`training`

:ref:`validation`

:ref:`empirical`

While it might be possible to run smaller tests on a laptop, it is generally advisable to seek out a high performance computing cluster, particularly for the simulation step.




.. _simulation:

*************   
1. Simulation
*************

Although ``disperseNN2`` does not actually have any simulation-related capabilities, it relies on simulated training data. Therefore, we provide some template code and tips for generating training data using ``SLiM``. However, the ideal analysis will tailor the simulation step to take advantage of realistic information from your study system. For information on how to implement population genetic simulations, check out the extensive ``SLiM`` manual (give link, here).

Our simulations use the script SLiM_recipes/bat20.slim. This simualation model is described in detail in Battey et al., 2020 (https://doi.org/10.1534/genetics.120.303143), but it is a continuous space model where the mother-offspring distance is N(0,sigma)—Gaussian distributed with mean zero and standard deviation sigma—in both the x and y dimensions. As a demonstration, see the below example command.




.. code-block:: bash

		mkdir temp_wd
		
		slim \
		-d SEED=12345 \
		-d sigma=0.2 \
		-d K=5 \
		-d mu=0 \
		-d r=1e-8 \
		-d W=50 \
		-d G=1e8 \
		-d maxgens=100 \
		-d OUTNAME="'temp_wd/TreeSeqs/my_sequence'" \
		SLiM_recipes/bat20.slim \
		# Note the two sets of quotes around the output name
		
Command line arguments are passed to ``SLiM`` using the `-d` flag followed by the variable name as it appears in the recipe file.

- ``SEED`` - a random seed to reproduce the simulation results.
- ``sigma`` - the dispersal parameter.
- ``K`` - carrying capacity.
- ``mu`` - per base per genertation mutation rate.
- ``r`` -  per base per genertation recombination rate.
- ``W`` - the height and width of the geographic spatial boundaries.
- ``G`` - total size of the simulated genome.
- ``maxgens`` - number of generations to run simulation.
- ``OUTNAME`` - prefix to name output files.

After running ``SLiM`` for a fixed number of generations, the simulation is still not complete, as many trees will likely not have coalesced still. Next we will finish, or "recapitate", the tree sequences. We recommend recapitating up front, as training is prohibitively slow if you try to recapitate on-the-fly. Below is an example recapitation command in python:

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

The preprocessing step converts the tree sequences output by ``SLiM`` into numpy arrays that are faster to read during training.

Side note: The reason ``disperseNN2`` does so much reading "on-the-fly" during training is to avoid loading thousands of tree sequences into memory at once;
the memory required for this would be significant, and unnecessary since numpy arrays can be read and released from memory sufficiently fast.

The preprocessing step can be parallelized to some extent: a single command preprocesses all simulations serially by taking one sample of genotypes from each dataset, so independent commands can be used with different random number seeds to take multiple, pseudo-independent samples from each simulation.

A basic preprocessing command looks like:

.. code-block:: bash
		
		python disperseNN2/disperseNN2.py \
		--out $box"_"n$n"_"$snps"snps_"preprocess_ONESIG \
		--num_snps $snps \
		--max_epochs 1000 \
		--validation_split 0.2 \
		--batch_size 1 \
		--threads 1 \
		--n $n \
		--seed XX \
		--num_samples 50 \
		--edge_width 3 \
		--learning_rate 1e-4 \
		--tree_list $trees \
		--target_list $targets \
		--preprocess






.. _training:

***********
3. Training
***********

Below is an example command for the training step.
This example uses tree sequences as input (again, feel free to kill this command).

.. code-block:: bash

		python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"$num_snps"snps_"preprocess_ONESIG --num_snps $num_snps --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads $threads --n $n --seed $id --train --learning_rate 1e-4 --preprocessed --pairs $pairs --pairs_encode $pairs2 --pairs_estimate $pairs3 --gpu -1 > Boxes$box"_"n$n"_"$num_snps"snps_"preprocess_ONESIG/out_one_sig.$id.txt_n$n"_"$pairs"pair_"$DATE



- ``tree_list``: list of paths to the tree sequences. &#963; values and habitat widths are extracted directly from the tree sequence.
- ``mutate``: add mutations to the tree sequence until the specified number of SNPs are obtained (5,000 in this case, specified inside the training params file).
- ``min_n``: specifies the minimum sample size.
- ``max_n``: paired with ``min_n`` to describe the range of sample sizes to drawn from. Set ``min_n`` equal to ``max_n`` to use a fixed sample size.
- ``edge_width``: this is the width of edge to 'crop' from the sides of the habitat. In other words, individuals are sampled ``edge_width`` distance from the sides of the habitat.
- ``sampling_width``: samples individuals from a restricted sampling window with width between 0 and 1, in proportion to the habitat width, after excluding edges.
- ``num_snps``: the number of SNPs to use as input for the CNN.
- ``repeated_samples``: this is the number of repeated draws of ``n`` individuals to take from each tree sequence. This let's us get away with fewer simulations.
- ``batch_size``: for the data generator. We find that batch_size=40 works well if the training set is larger.
- ``threads``: number of threads to use for multiprocessing during the data generation step.
- ``max_epochs``: maximum number of epochs to train for.
- ``seed``: random number seed.
- ``out``: output prefix.

This run will eventually print the training progress to stdout, while the model weights are saved to ``temp_wd/out1_model.hdf5``.

Also, this example command is small-scale; in practice, you will need a training set of maybe 50,000, and you will want to train for longer than 10 epochs.






.. _validation:

*************
4. Validation
*************

If you want to predict sigma from simulated tree sequences output by ``SLiM``, a predict command like the below one can be used (should take <30s to run). Each command line flag is described in the preceding examples(??)


.. code-block:: bash

		python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess_ONESIG --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 1 --threads 1 --n $n --seed $id --num_samples 50 --predict --learning_rate 1e-4 --preprocessed --pairs $pairs --load_weights Boxes105_106_n23_preprocess_ONESIG/out140_boxes105_noProj_model.hdf5 --num_pred 100 --gpu_index -1                                                                                                                              


Similar to the earlier prediction example, this will generate a file called `temp_wd/out_treeseq_predictions.txt` containing:

.. code-block:: bash

		Examples/TreeSeqs/output_2_recap.trees 0.5914545564 0.6582331812
		Examples/TreeSeqs/output_3_recap.trees 0.3218814158 0.3755014635
		Examples/TreeSeqs/output_1_recap.trees 0.3374337601 0.4073884732
		Examples/TreeSeqs/output_5_recap.trees 0.2921853737 0.2047981935
		Examples/TreeSeqs/output_4_recap.trees 0.277020769 0.3208989912


Here, the second and third columns contain the true and predicted sigma; for each simulation.









.. _empirical:

************************
5. Empirical predictions
************************

For predicting with empirical data, the command will be slightly different: instead of a list of tree sequences (and targets?), a new flag is given, --empirical, which is a prefix for two files: a VCF and a table of lat and long. The lat and longs get projected onto a flat 2D map using ____.


.. code-block:: bash

                python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess_ONESIG --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 1 --threads 1 --n $n --seed $id --num_samples 50 --predict --learning_rate 1e-4 --preprocessed --pairs $pairs --load_weights Boxes105_106_n23_preprocess_ONESIG/out140_boxes105_noProj_model.hdf5 --num_pred 100 --gpu_index -1

