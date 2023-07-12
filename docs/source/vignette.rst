Vignette: example workflow
==========================


This vignette shows a complete pipeline for a small application of ``disperseNN2`` including instructions for the intermediate data-organizing steps. Some details referenced in the below vignette, e.g., descriptions of command line flags, are explained under :doc:`usage`.



**Table of contents:**

:ref:`vignette_simulation`

:ref:`vignette_preprocessing`

:ref:`vignette_training`

:ref:`vignette_validation`

:ref:`vignette_empirical`

     

.. _vignette_simulation:

1. Simulation
-------------

For this demonstration we will analyze a population of *Internecivus raptus*. Let's assume we have independent estimates from previous studies for several parameters:

- the width of the species range is :math:`78` km
- population density is 2.5 individuals per km\ :math:`^2`
- recombination rate is 1e-8 crossovers per bp per generation

With values for these nuisance parameters in hand we can design custom training simulations for inferring :math:`\sigma`. If our a priori expectation for :math:`\sigma` in this species is somewhere between 0.4 and 6, we will simulate dispersal rates in this range.

Below is some bash code to run the simulations using ``square.slim``. 

.. code-block:: bash
   :linenos:

   mkdir -p temp_wd/vignette/TreeSeqs
   mkdir -p temp_wd/vignette/Targets		
   sigmas=$(python -c 'from scipy.stats import loguniform; print(*loguniform.rvs(0.4,6,size=100))')
   for i in {1..100}
   do
       sigma=$(echo $sigmas | awk -v var="$i" '{print $var}')
       echo "slim -d SEED=$i -d sigma=$sigma -d K=2.5 -d r=1e-8 -d W=78 -d G=1e8 -d maxgens=1000 -d OUTNAME=\"'temp_wd/vignette/TreeSeqs/output'\" SLiM_recipes/square.slim" >> temp_wd/vignette/sim_commands.txt
       echo $sigma > temp_wd/vignette/Targets/target_$i.txt
       echo temp_wd/vignette/Targets/target_$i.txt >> temp_wd/vignette/target_list.txt
   done
   num_threads=2 # change to number of available cores
   parallel -j $num_threads < temp_wd/vignette/sim_commands.txt

Breaking down this pipeline one line at a time:

- L1 creates a new folder for the simulation output. The base folder ``temp_wd`` will contain all output from the current vignette.
- L2 creates another folder for the training targets.
- L3 draws random :math:`\sigma`\'s from a log-uniform distribution.
- L7 builds individual commands for simulations.
- L8 saves each :math:`\sigma` to it's own file.
- L9 creates a list of filepaths to the targets.
- L12 runs the simulation commands. If multiple cores are available, the number of threads used for this vignette can be increased (L11) to speed things up. In a real application, simulations should be distributed across many jobs on a computing cluster.

And to recapitate the tree sequences output by ``SLiM``:

.. code-block:: bash

		for i in {1..100}
		do
		    echo "python -c 'import tskit,msprime; \
		                     ts=tskit.load(\"temp_wd/vignette/TreeSeqs/output_$i.trees\"); \
				     Ne=len(ts.individuals()); \
				     demography = msprime.Demography.from_tree_sequence(ts); \
				     demography[1].initial_size = Ne; \
				     ts = msprime.sim_ancestry(initial_state=ts, recombination_rate=1e-8, demography=demography, start_time=ts.metadata[\"SLiM\"][\"cycle\"],random_seed=$i,); \
				     ts.dump(\"temp_wd/vignette/TreeSeqs/output_$i"_"recap.trees\")'" \
		    >> temp_wd/vignette/recap_commands.txt
		    echo temp_wd/vignette/TreeSeqs/output_$i"_"recap.trees >> temp_wd/vignette/tree_list.txt
		done   
		parallel -j $num_threads < temp_wd/vignette/recap_commands.txt








		



.. _vignette_preprocessing:

2. Preprocessing
----------------

Next, we need to preprocess the input for ``disperseNN2``. But first we need to clean up our *I. raptus* metadata.

Let's pretend we want to take a subset of individuals from a particular geographic region, the "Scotian Shelf-East" region. Below is an example command that might be used to parse and reformat the metadata, but these steps will vary depending on the idiosyncracies of your particular dataset. 

.. code-block:: bash

		cat Examples/VCFs/iraptus_meta_full.txt | grep "Scotian Shelf - East" | sed s/"\t"/,/g > temp_wd/vignette/iraptus.csv

We provide a simple python script for subsetting a VCF for a particular set of individuals, which also filters indels and non-variant sites.

.. code-block:: bash

		python Empirical/subset_vcf.py Examples/VCFs/iraptus_full.vcf.gz temp_wd/vignette/iraptus.csv temp_wd/vignette/iraptus.vcf 0 1
		gunzip temp_wd/vignette/iraptus.vcf.gz

Last, build a .locs file:

.. code-block:: bash

		count=$(zcat temp_wd/vignette/iraptus.vcf.gz | grep -v "##" | grep "#" | wc -w)
		for i in $(seq 10 $count); do id=$(zcat temp_wd/vignette/iraptus.vcf.gz | grep -v "##" | grep "#" | cut -f $i); grep -w $id temp_wd/vignette/iraptus.csv; done | cut -d "," -f 4,5 | sed s/","/"\t"/g > temp_wd/vignette/iraptus.locs

This filtering results in 1951 SNPs from 95 individuals. We will take 10 repeated samples from each tree sequence, to get a total of 1,000 training datasets (100 tree sequences :math:`\times` 10 samples from each). Our strategy for doing this involves 10 different preprocess commands, each with a different random number seed, which can be run in parallel.

.. code-block:: bash
		
		for i in {1..10}
		do
		    echo "python disperseNN2.py \
		                 --out temp_wd/vignette/output_dir \
				 --preprocess \
				 --num_snps 1951 \
				 --n 95 \
				 --seed $i \
				 --tree_list temp_wd/vignette/tree_list.txt \
				 --target_list temp_wd/vignette/target_list.txt \
				 --empirical temp_wd/vignette/iraptus" \
		    >> temp_wd/vignette/preprocess_commands.txt
		done
		parallel -j $num_threads < temp_wd/vignette/preprocess_commands.txt










   


		       


.. _vignette_training:

3. Training
-----------

In the below ``disperseNN2`` training command, we set the number of pairs to 1000; this is the number of pairs of individuals from each training dataset that are included in the analysis, and we chose 1000 in order to fit within available memory. The maximum number of pairs with 95 individuals would have been 4465. We've found that using 100 for ``--pairs_encode`` and ``--pairs_estimate`` works well, while reducing memory requirements. Don't forget to tack on the ``--gpu`` flag if GPUs are available.

.. code-block:: bash

                python disperseNN2.py \
                       --out temp_wd/vignette/output_dir \
                       --train \
                       --preprocessed \
                       --num_snps 1951 \
                       --max_epochs 20 \
                       --validation_split 0.2 \
                       --batch_size 10 \
                       --threads 1 \
                       --seed 12345 \
                       --n 95 \
                       --learning_rate 1e-4 \
                       --pairs 1000 \
                       --pairs_encode 100 \
                       --pairs_estimate 100 \
                       > temp_wd/vignette/output_dir/training_history.txt





		       






.. _vignette_validation:

4. Validation
-------------

Next, we will validate the trained model on simulated test data. In a real application you should hold out datasets from training, but we haven't updated the disperseNN code to do this yet.

.. code-block:: bash

                python disperseNN2.py \
                       --out temp_wd/vignette/output_dir \
                       --predict \
                       --preprocessed \
                       --num_snps 1951 \
                       --batch_size 10 \
                       --threads 1 \
                       --n 95 \
                       --seed 12345 \
                       --pairs 1000 \
                       --pairs_encode 100 \
                       --pairs_estimate 100 \
                       --load_weights temp_wd/vignette/output_dir/pwConv_12345_model.hdf5 \
                       --num_pred 100
		       
.. figure:: results.png
   :scale: 50 %
   :alt: results_plot

   Validation results after 100 epochs of training. True :math:`\sigma` is on the x-axis and predicted values are on the y-axis. The dashed line is :math:`x=y`.
		       
The results show that the training run was successful. Specifically, the predictions are near the expected values, meaning there is some signal for dispersal rate.

.. However, we are currently underestimating towards the larger end of the :math:`\sigma` range. This might be alleviated by using (i) a larger training set, (ii) more generatinos spatial, (iii) larger sample size, or (iv) or more SNPs.








.. _vignette_empirical:

5. Empirical application
------------------------

Since we are satisfied with the performance of the model on the held-out test set, we can finally predict σ in our empirical data.

.. code-block:: bash

		python disperseNN2.py \
                       --out temp_wd/vignette/output_dir \
		       --predict \
		       --empirical temp_wd/vignette/iraptus \
		       --num_snps 1951 \
		       --batch_size 10 \
		       --threads 1 \
		       --n 95 \
		       --seed 12345 \
                       --pairs 1000 \
		       --pairs_encode 100 \
                       --pairs_estimate 100 \
                       --load_weights temp_wd/vignette/output_dir/pwConv_12345_model.hdf5 \
                       --num_reps 10

The final empirical results are stored in: ``temp_wd/vignette/output_dir/empirical_12345_predictions.txt``.

.. code-block:: bash

		temp_wd/vignette/iraptus_0 4.2500737306
		temp_wd/vignette/iraptus_1 4.491881125
		temp_wd/vignette/iraptus_2 4.5922921796
		temp_wd/vignette/iraptus_3 4.6481258415
		temp_wd/vignette/iraptus_4 4.8540334937
		temp_wd/vignette/iraptus_5 5.1935628625
		temp_wd/vignette/iraptus_6 4.3853448496
		temp_wd/vignette/iraptus_7 3.9447901711
		temp_wd/vignette/iraptus_8 4.2443830458
		temp_wd/vignette/iraptus_9 4.2815212007


**Interpretation**.
Sigma is the SD of the gaussian dispersal kernel. The distance to a random parent is root-2 * sigma.
We trained with only 100 generations spatial, hence the estimate reflects demography in the recent past.










To Do:
- find some data that are better than halibut
- random number seeds currently not working
- separate training and test sims internally, automatically, using disperseNN.
B
