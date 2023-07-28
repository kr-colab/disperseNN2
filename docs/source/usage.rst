




.. _usage:

Usage
-----



.. _install:

Install
^^^^^^^

We recommend creating a new conda environment to stay organized:

.. code-block:: console

		(.venv) $ conda create -n disperseNN2 python=3.9 --yes
                (.venv) $ conda activate disperseNN2


Then install ``disperseNN2`` using pip:

.. code-block:: console

                (.venv) $ pip install disperseNN2

``disperseNN2`` should run fine on just CPUs. But if you have GPUs available, see our :ref:`gpus` installation tips.









Workflow
^^^^^^^^
This section describes the command line flags associated with each step in the workflow; for a complete, worked example, see :doc:`vignette`.

A typical ``disperseNN2`` workflow involves four steps:



:ref:`preprocessing`

:ref:`training`

:ref:`validation`

:ref:`empirical`



     



.. _preprocessing:

****************
1. Preprocessing
****************

``disperseNN2`` trains on simulated data (see :ref:`simulation`) which produce output in the form of tree sequences.
The "preprocessing step" adds mutations to the tree sequences, takes a sample of individuals, and then saves the genotypes and sample locations in numpy arrays.
Doing these steps up front instead of during training is faster.

A basic preprocessing command looks like:

.. code-block:: console
		
		(.venv) $ disperseNN2 \
                >             --out <path> \
		>             --seed <int> \
		>	      --preprocess \
		>             --n <int> \
		>	      --num_snps <int> \
		>	      --tree_list <path> \
		>	      --target_list <path> \
		>	      --empirical <path> \
		>	      --hold_out <int>

- ``--out``: output directory
- ``--preprocess``: this flag tells ``disperseNN2`` to preprocess the training data
- ``--n``: sample size
- ``--num_snps``: the number of SNPs to use as input for the CNN
- ``--tree_list``: path to a list of filepaths to the tree sequences
- ``--target_list``: path to list of filepaths to .txt files with the target values
- ``--empirical``: prefix for the empirical locations. This includes the path, but without the filetype suffix, ".locs".
- ``--hold_out``: number of tree sequences to hold out from training, to be used for testing later on
- ``--seed``: random number seed

Simulated individuals are sampled near the empirical sample locations: a table with one row per individual, with latitude and longitude tab-separated. Our strategy involves first projecting the geographic coordinates for each location onto a 2D surface. By default, the projected locations are repositioned to new, random areas of the training map before sampling individuals from those locations; this is making the assumption that the true habitat range is unknown and we want our predictions to be invariant to the position of the sampling area within the greater species distribution.

.. Last, the spatial coordinates are rescaled to :math:`(0,1)`, preserving aspect ratio, before being shown to the neural network as input.
  
The preprocessed data are saved in the directory specified by ``--out``; ``disperseNN2`` will look in this folder for inputs and outputs in the following steps.







.. _training:

***********
2. Training
***********

..
    DEV:
        Preprocessing and training commands to get the training data, after simulating as in the vignette
	python disperseNN2.py                  --out temp_wd/vignette/output_dir_n10                  --seed 12345                  --preprocess                  --num_snps 1951                  --n 10                  --tree_list temp2                  --target_list temp1                  --empirical Examples/VCFs/halibut                  --hold_out 10
	python disperseNN2.py                --out Examples/Preprocessed                --seed 67890                --train                --num_snps 1951                --max_epochs 50                --validation_split 0.2                --batch_size 10                --threads 1                --n 10                --pairs 45                --pairs_encode 45                --pairs_estimate 45                --gpu 2



Below is what a command looks like for the training step. 

.. code-block:: console

		(.venv) $ disperseNN2 \
		>             --out <path> \
		>             --seed <int> \
		>	      --train \
		>	      --max_epochs <int> \
		>	      --validation_split <float> \
		>	      --batch_size <int> \
		>	      --threads <int> \
		>	      --pairs <int> \
		>	      --pairs_encode <int> \
		>             --threads <int> \
		>	      --gpu <int> \

- ``--train``: tells ``disperseNN2`` to train a neural network
- ``--max_epochs``: maximum number of epochs to train for.
- ``--validation_split``: the proportion of training data held out for validation between batches for hyperparameter tuning. We use 0.2.
- ``--batch_size``: we find that batch_size=10 works well.
- ``--threads``: number of threads to use during training. 
- ``--pairs``: the total number of pairs to include in the analysis. Defaults to all pairs.
- ``--pairs_encode``: the number of pairs to include in the gradient in the encoder portion of the neural network. Default: all pairs.
- ``--threads``; the number of threads to use. This works pretty well for speeding up training or prediction. 40-50 CPUs approximates the speed of one GPU.
- ``--gpu``: as an integer, specifies the GPU index (e.g., 0, 1, etc). "any" means take any available gpu. -1 means no GPU.

This command will print the training progress to stdout.
The model weights are saved to ``<out>/Train/disperseNN2_<seed>_model.hdf5``.
A single thread should be sufficient for reading preprocessed data, but we found that between 2 and 10 threads speeds up training.

After training has completed (or has been interrupted), the training history can be visualized using a ``disperseNN2`` functionality:

.. code-block:: console

                (.venv) $ python disperseNN2.py --plot_history <path_to_training_history>

..		
   .. figure:: training_usage.png
   :scale: 50 %
   :alt: training_plot

   Plot of training history. X-axis the	training iteration, and	Y-axis is mean squared error.



		






.. _validation:

*************
3. Validation
*************

If you want to predict :math:`\sigma` from simulated data, a predict command like the below one can be used. 

.. code-block:: console

		(.venv) $ disperseNN2 \
		>             --out <path> \
		>             --seed <int> \
		>	      --predict \
		>	      --batch_size <int> \
		>	      --num_pred <int>

- ``--predict``: tells ``disperseNN2`` to perform predictions
- ``--num_pred``: number of datasets to predict with.

This will generate a file called ``<out>/Test/predictions_<seed>.txt`` containing true and predicted :math:`\sigma` for each simulation.









.. _empirical:

************************
4. Empirical prediction
************************

For predicting with empirical data, we provide the program with (1) a .vcf and (2) a .locs file (mentioned above, with preprocessing). The order of individuals in the .vcf needs to match that of the .locs file. SNPs should be minimally filtered to exclude indels, multi-allelic sites, and maybe low-confidence variant calls; however, low-frequency SNPs should be left in as these are informative about demography.

.. code-block:: console

                (.venv) $ disperseNN2 \
                >             --out <path> \
		>	      --seed <int> \		       
		>	      --predict \
		>	      --empirical <path> \
		>	      --num_reps <int>

- ``--empirical``: prefix for the empirical data that is shared for both the .vcf and .locs files. This includes the path, but without the filetype suffix. 
- ``--num_reps``: specifies how many bootstrap replicates to perform. Each replicate takes a random draw of num_snps SNPs from the VCF.

The output is in kilometers and can be found in ``<out>/empirical_<seed>.txt``:

..
		(.venv) $ cat Examples/Preprocessed/empirical_67890.txt
		Examples/VCFs/halibut rep0 2.4848595098
		Examples/VCFs/halibut rep1 2.2881405623
		Examples/VCFs/halibut rep2 1.8599958634
		Examples/VCFs/halibut rep3 2.4091420017
		Examples/VCFs/halibut rep4 2.3767512964






.. _simulation:

Simulation
^^^^^^^^^^

Although ``disperseNN2`` is not used for running simulations, it relies on simulated training data. Therefore, we provide some template code for generating training data. However, the ideal analysis will tailor the simulation step to take advantage of realistic information about your particular study system. For information on how to implement population genetic simulations, check out the `SLiM manual <http://benhaller.com/slim/SLiM_Manual.pdf>`_.

The simulation script we use to train ``disperseNN2`` is ``SLiM_recipes/square.slim``. This is a continuous space model where mother-offspring dispersal is :math:`N(0,\sigma)` in both the :math:`x` and :math:`y` dimensions. Other details of the model are described in `Battey et al. 2020 <https://doi.org/10.1534/genetics.120.303143>`_. Below is the code for the simulation:


.. code-block::

   initialize() {

       setSeed(SEED);
       print( c("new seed:",getSeed()) );
       initializeSLiMModelType("nonWF");
       initializeSLiMOptions(dimensionality="xy");
       initializeTreeSeq(); 
       defineConstant("SD", sigma);  // sigma_D, the dispersal distance
       defineConstant("SI", sigma);  // sigma_I, the spatial interaction distance
       defineConstant("SM", SI);  // sigma_M, the mate choice distance
       defineConstant("L", 4);    // mean lifetime at stationarity
       defineConstant("FECUN", 1/L); // mean fecundity
       defineConstant("RHO", FECUN/((1+FECUN) * K)); // constant in spatial competition function
       initializeMutationType("m1", 0.5, "g", 0.0, 2);
       initializeGenomicElementType("g1", m1, 1.0);
       initializeGenomicElement(g1, 0, G-1);
       initializeMutationRate(0);
       initializeRecombinationRate(r);  
       initializeInteractionType(1, "xy", reciprocal=T, maxDistance=SI * 3);
       i1.setInteractionFunction("n", 1.0/(2*PI*SI^2), SI);
       initializeInteractionType(2, "xy", reciprocal=T, maxDistance=SM * 3);
       i2.setInteractionFunction("n", 1.0/(2*PI*SM^2), SM);
   }

   reproduction() {
       mate = i2.drawByStrength(individual, 1);
       if (mate.size()) {
           nOff = rpois(1, FECUN);
           for (i in seqLen(nOff)) {
               pos = individual.spatialPosition + rnorm(2, 0, SD);
               if (p1.pointInBounds(pos)) {
                   offspring = subpop.addCrossed(individual, mate);
		   offspring.setSpatialPosition(pos);
	       }
           }
       }
       return;
   }

   1 early() {
       sim.addSubpop("p1", asInteger(K * W * W));
       p1.setSpatialBounds(c(0, 0, W, W));
       for (ind in p1.individuals) {
           ind.setSpatialPosition(p1.pointUniform());
       }
       i1.evaluate(p1);
   }

   early() {
       i1.evaluate(p1);
       inds = p1.individuals;
       competition = i1.localPopulationDensity(inds);
       inds.fitnessScaling = 1/(1 + RHO * competition);    
   }

   1: late() {
       // to be ready for mate choice
       i2.evaluate(p1);
   }

   1: late() {
       print(c("Finished generation", sim.cycle, "; N=", p1.individualCount));
       if (p1.individualCount == 0){
           catn("Population died.");
           sim.simulationFinished();
       }
       else{
	   // end after maxgens
	   if (sim.cycle == maxgens){
               sim.treeSeqOutput(paste(c(OUTNAME,"_",SEED,".trees"), sep=""));
               catn("Done.");
               sim.simulationFinished();
	   }
       }
   }

   999999999 late() {} 


If you want to run the simulation, save the above script as ``square.slim``, and install ``SLiM``:

   
.. code-block:: console

                (.venv) $ conda install slim==4.0.1 -c conda-forge

Below is an example command using this script:
		
.. code-block:: console

		(.venv) $ slim -d SEED=<int> \
                >              -d sigma=<float> \     
		> 	       -d K=<int> \
		>	       -d r=<float> \
		>	       -d W=<int> \
		>	       -d G=<int> \
		>	       -d maxgens=<int> \
		>	       -d OUTNAME="'<path>'" \
		>	       square.slim
		
Command line arguments are passed to ``SLiM`` using the ``-d`` flag followed by the variable name as it appears in the recipe file.

- ``SEED``: a random seed to reproduce the simulation results.
- ``sigma``: the dispersal parameter.
- ``K``: carrying capacity. Note: the carrying capacity in this model, K, corresponds roughly to density, but the actual density will vary depending on the model, and will fluctuate a bit over time.
- ``r``:  per base per genertation recombination rate.
- ``W``: the height and width of the geographic spatial boundaries.
- ``G``: total size of the simulated genome.
- ``maxgens``: number of generations to run simulation.
- ``OUTNAME``: prefix to name output files. Note the two sets of quotes around the output name

In the ``disperseNN2`` paper we ran 100,000 spatial generations. After running ``SLiM`` for a fixed number of generations, the simulation is still not complete, as many trees will likely not have coalesced still. Next you will need to finish, or "recapitate", the tree sequences. We recommend recapitating at this early stage, before training, as training can be prohibitively slow if you recapitate on-the-fly. The below code snippet in python can be used to recapitate a tree sequence:

.. code-block:: pycon

		>>> import tskit,msprime
		>>> ts=tskit.load("<prefix>.trees")
		>>> Ne=len(ts.individuals())
		>>> demography = msprime.Demography.from_tree_sequence(ts)
		>>> demography[1].initial_size = Ne
		>>> ts = msprime.sim_ancestry(initial_state=ts, recombination_rate=<r>, demography=demography, start_time=ts.metadata["SLiM"]["cycle"],random_seed=12345)
		>>> ts.dump("<prefix>_recap.trees")

Where ``prefix`` is a path to a tree sequence excluding ".trees", and ``r`` is the recombination rate.

.. note::

   Here, we have assumed a constant demographic history. If an independently inferred demographic history for your species is available, or if you want to explore different demographic histories, the recapitation step is a good place for implementing these changes. For more information see the `msprime docs <https://tskit.dev/msprime/docs/stable/ancestry.html#demography>`_.

For planning the total number of simulations, consider the following. If the simulations explore a large parameter space, e.g. more than	one or two free	parameters, then larger training sets may be required.	In our paper, we used a training set of 50,000—--but, this is number may depend on the training distribution, Last, don't forget to run extra simulations (e.g., 100 or 1000) to validate your model with post training.

Simulation programs other than ``SLiM`` could be used in theory. The only real requirements of ``disperseNN2`` regarding training data are: genotypes are in a 2D array, the corresponding sample locations are in a table with two columns, and the target values are saved in individual files; all as numpy arrays. 
		
