Vignette: example workflow
==========================


This vignette shows a complete pipeline for a small application of ``disperseNN2`` including instructions for the intermediate data-organizing steps. For details about individual command line flags, see :doc:`usage`.



**Table of contents:**

:ref:`vignette_simulation`

:ref:`vignette_preprocessing`

:ref:`vignette_training`

:ref:`vignette_validation`

:ref:`vignette_empirical`

     

.. _vignette_simulation:

1. Simulation
-------------

For this demonstration we will analyze a population of Internecivus raptus. Let's assume we have independent estimates from previous studies for the size of the species range and the population density: :math:`48 \times 48` km\ :math:`^2`, and 6 individuals per km\ :math:`^2`, respectively. With values for these nuisance parameters in hand we can design custom training simulations for inferring :math:`\sigma`. If our a priori expectation for :math:`\sigma` in this species is somewhere between 0.2 and 1.5, we will simulate dispersal rates in this range.

Below is some bash code to run the simulations using ``bat20.slim``. 

.. code-block:: bash
   :linenos:

   mkdir -p temp_wd/vignette/TreeSeqs
   mkdir -p temp_wd/vignette/Targets		
   sigmas=$(python -c 'from scipy.stats import loguniform; print(*loguniform.rvs(0.2,1.5,size=100))')
   for i in {1..100}
   do
       sigma=$(echo $sigmas | awk -v var="$i" '{print $var}')
       echo "slim -d SEED=$i -d sigma=$sigma -d K=6 -d mu=0 -d r=1e-8 -d W=48 -d G=1e8 -d maxgens=100 -d OUTNAME=\"'temp_wd/vignette/TreeSeqs/output'\" SLiM_recipes/bat20.slim" >> temp_wd/vignette/sim_commands.txt
       echo $sigma > temp_wd/vignette/Targets/target_$i.txt
       echo temp_wd/vignette/Targets/target_$i.txt >> temp_wd/vignette/target_list.txt
   done
   parallel -j 2 < temp_wd/vignette/sim_commands.txt

Breaking down this pipeline one line at a time:

- L1 creates a new folder where the output from the vignette will be saved.
- L2 creates another folder for the training targets.
- L3 draws random :math:`\sigma`\'s from a log-uniform distribution.
- L7 builds individual commands for simulations.
- L8 saves each :math:`\sigma` to it's own file.
- L9 creates a list of filepaths to the targets.
- L11 runs the simulation commands. Here, we use GNU ``parallel`` with 2 threads; if multiple cores are available, the number of threads used for this vignette can be increased to speed things up. In a real application, simulations should probably be distributed across many jobs on computing cluster.

And to recapitate the tree sequences output by ``SLiM``:

.. code-block:: bash

		for i in {1..100}
		do
		    echo "python -c 'import tskit,msprime; ts=tskit.load(\"temp_wd/vignette/TreeSeqs/output_$i.trees\"); Ne=len(ts.individuals()); demography = msprime.Demography.from_tree_sequence(ts); demography[1].initial_size = Ne; ts = msprime.sim_ancestry(initial_state=ts, recombination_rate=1e-8, demography=demography, start_time=ts.metadata[\"SLiM\"][\"cycle\"],random_seed=$i,); ts.dump(\"temp_wd/vignette/TreeSeqs/output_$i"_"recap.trees\")'" >> temp_wd/vignette/recap_commands.txt
		    echo temp_wd/vignette/TreeSeqs/output_$i"_"recap.trees >> temp_wd/vignette/tree_list.txt
		done   
		parallel -j 2 < temp_wd/vignette/recap_commands.txt







.. _vignette_preprocessing:

2. Preprocessing
----------------

Next, we preprocess the input for ``disperseNN2``. Assume we have a sample of 97 individuals from different locations, and 25,000 SNPs.

.. code-block:: bash
		
		python disperseNN2.py \
		       --out temp_wd/vignette/output_dir \
                       --preprocess \
                       --num_snps 25000 \
                       --n 97 \
                       --seed 1 \
                       --edge_width 1.5 \
                       --tree_list temp_wd/vignette/tree_list.txt \
                       --target_list temp_wd/vignette/target_list.txt

.. note::

   Here we chose to sample away from the habitat edges by 1.5km. This is because the simulation model artifically reduces survival probability near the edges, within distance :math:`\sigma`, roughly. Since the largest :math:`\sigma` we explored is 1.5, we simply cropped away this width from each edge.





		       


.. _vignette_training:

3. Training
-----------

And the training step:

.. code-block:: bash

                python disperseNN2.py \
                       --out temp_wd/vignette/output_dir \
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
                       > temp_wd/vignette/output_dir/training_history.txt \
		       # do we need the "n" flag?


		       

.. _vignette_validation:

4. Validation
-------------

.. code-block:: bash

                python disperseNN2.py \
                       --out temp_wd/vignette/output_dir \
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
                       --load_weights temp_wd/vignette/output_dir/out_12345_model.hdf5 \
                       --num_pred 5










.. _vignette_empirical:

5. Empirical application
------------------------

.. code-block:: bash

		python disperseNN2.py \
                       --out temp_wd/vignette/output_dir \
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
                       --load_weights temp_wd/vignette/output_dir/out_12345_model.hdf5 \
                       --num_pred 1







