Vignette: example workflow
==========================


This vignette shows a complete pipeline for a small application of ``disperseNN2``. Some details referenced in the below vignette, e.g., descriptions of command line flags, are explained under :doc:`usage`.
We recommend using a machine with 10s of CPUs for the below analysis.




**Table of contents:**

:ref:`vignette_simulation`

:ref:`vignette_preprocessing`

:ref:`vignette_training`

:ref:`vignette_validation`

:ref:`vignette_empirical`

:ref:`google_colab_notebook`

     
.. _vignette_simulation:

1. Simulation
-------------
A complete workflow would include running simulations that approximate the life history and species range of a focal organism.
Here we will use simulations from the ``SLiM`` package to generate training data for ``disperseNN2``. 
The simulations use a simple model of a species with a square range and a Gaussian dispersal kernel.

To start with: create a new working directory and install ``SLiM`` if you haven't yet:

.. code-block:: console

                (.venv) $ mkdir temp_wd
                (.venv) $ cd temp_wd
                (.venv) $ conda install slim==4.0.1 -c conda-forge


For this demonstration we will analyze a sample of 95 individuals from a population of `Internecivus raptus <https://en.wikipedia.org/wiki/Xenomorph>`_. Let's assume we have independent estimates from previous studies for several parameters:

- the width of the species range is 78 km
- population density is 2.5 individuals per km\ :math:`^2`
- recombination rate is 1e-8 crossovers per bp per generation

With values for these nuisance parameters in hand we can design custom training simulations for inferring :math:`\sigma`. If our a priori expectation for :math:`\sigma` in this species is somewhere between 0.4 and 6, we will simulate dispersal rates in this range. 100 training simulations should suffice for this demonstration, plus 100 more for testing, so we need 200 total simulations.		

Next copy over the `square.slim script <https://github.com/andrewkern/disperseNN2/blob/main/SLiM_recipes/square.slim>`_, introduced in the :ref:`simulation` section of the docs. Below is some bash code for pipelining the simulations.

.. code-block:: console                         
                :linenos:                       

                (.venv) $ mkdir -p vignette/TreeSeqs
                (.venv) $ mkdir -p vignette/Targets
		(.venv) $ sigmas=$(python -c 'from scipy.stats import loguniform; import numpy; numpy.random.seed(seed=12345); print(*loguniform.rvs(0.4,6,size=200))')
                (.venv) $ for i in {1..200}; do \
                >             sigma=$(echo $sigmas | awk -v var="$i" '{print $var}'); \
		>             echo "slim -d SEED=$i -d sigma=$sigma -d K=2.5 -d r=1e-8 -d W=78 -d G=1e8 -d maxgens=1000 -d OUTNAME=\"'vignette/TreeSeqs/output'\" square.slim" >> vignette/sim_commands.txt; \
		>             echo $sigma > vignette/Targets/target_$i.txt; \
		>             echo vignette/Targets/target_$i.txt >> vignette/target_list.txt; \
		>         done

Breaking down this pipeline one line at a time:

- L1 creates a new folder for the simulation output. The base folder ``vignette`` will contain all output from the current vignette.
- L2 creates another folder for the training targets.
- L3 draws random :math:`\sigma`\'s from a log-uniform distribution.
- L6 builds individual commands for simulations.
- L7 saves each :math:`\sigma` to it's own file.
- L8 creates a list of filepaths to the targets.

.. note::

   The above example used only 1,000 spatial generations; this strategy should be used with caution because this can affect how the output is interpreted. In addition, isolation-by-distance is usually weaker with fewer spatial generations which reduces signal for dispersal rate. In the paper we used 100,000 spatial generations.

The below command runs the simulations. The number of simulations run in parallel can be adjusted with ``num_threads``:

.. code-block:: console

                (.venv) $ num_threads=1 # change to number of available cores
                (.venv) $ parallel -j $num_threads < vignette/sim_commands.txt
  
The longest of these simulations are expected to take over an hour.
Therefore, at this point we offer three options:
option (A) is to wait on the simulations to finish.
If you are feeling impatient, you may instead (B) download a .tar file (``wget http://sesame.uoregon.edu/~chriscs/output_dir.tar.gz;`` ``tar -xf output_dir.tar.gz -C vignette/``)  with the simulated and pre-processed data and skip to the :ref:`vignette_training` section.
Or, (C) check out our :ref:`google_colab_notebook` where the simulated data and GPUs are available.

The below command runs the simualtions (option A). The number of simulations run in parallel can be adjusted with ``num_threads``:

   
To recapitate the tree sequences output by ``SLiM``:

.. code-block:: console

		(.venv) $ for i in {1..200}; do \
		>             echo "python -c 'import tskit,msprime; \
		>                              ts=tskit.load(\"vignette/TreeSeqs/output_$i.trees\"); \
		>		               Ne=len(ts.individuals()); \
		>		               demography = msprime.Demography.from_tree_sequence(ts); \
		>		               demography[1].initial_size = Ne; \
		>		               ts = msprime.sim_ancestry(initial_state=ts, recombination_rate=1e-8, demography=demography, start_time=ts.metadata[\"SLiM\"][\"cycle\"],random_seed=$i,); \
		>		               ts.dump(\"vignette/TreeSeqs/output_$i"_"recap.trees\")'" \
		>             >> vignette/recap_commands.txt; \
		>             echo vignette/TreeSeqs/output_$i"_"recap.trees >> vignette/tree_list.txt; \
		>         done   
		(.venv) $ parallel -j $num_threads < vignette/recap_commands.txt











		



.. _vignette_preprocessing:

2. Preprocessing
----------------

Next, we need to preprocess the input for ``disperseNN2``. But before we do that we need to clean up our *I. raptus* metadata, because we will use the empirical sampling locations during preprocessing. Go ahead and clone our git repo which contains the empirical data we're analyzing, 

.. code-block:: console

                (.venv) $ wget https://raw.githubusercontent.com/kr-colab/disperseNN2/main/Examples/VCFs/iraptus_meta_full.txt -P vignette/
		(.venv) $ wget https://raw.githubusercontent.com/kr-colab/disperseNN2/main/Examples/VCFs/iraptus.vcf -P vignette/

Let's pretend we want to take a subset of individuals from a particular geographic region, the "Scotian Shelf-East" region. Below is an example command that might be used to parse and reformat the metadata, but these steps will vary depending on the idiosyncracies of your particular dataset. 

.. code-block:: console

		(.venv) $ cat vignette/iraptus_meta_full.txt | grep "Scotian Shelf - East" | sed s/"\t"/,/g > vignette/iraptus.csv


..
 We provide a simple script for subsetting a VCF for a particular set of individuals, which also filters indels and non-variant sites:

		(.venv) $ python Empirical/subset_vcf.py disperseNN2/Examples/VCFs/iraptus_full.vcf.gz vignette/iraptus.csv vignette/iraptus.vcf 0 1 12345
		(.venv) $ gunzip vignette/iraptus.vcf.gz
 The flags for ``Empirical/subset_vcf.py`` are:

 1. path to input vcf (gzipped)
 2. path to metadata (.csv)
 3. output name
 4. minimum read depth to retain a SNP (int)
 5. minimum proportion of samples represented to keep a SNP (float)
 6. random number seed (int)


    
Last, build a .locs file:

.. code-block:: console                                                                        
                                                                                            
                (.venv) $ count=$(cat vignette/iraptus.vcf | grep -v "##" | grep "#" | wc -w) 
                (.venv) $ for i in $(seq 10 $count); do \                                       
                >             id=$(cat vignette/iraptus.vcf | grep -v "##" | grep "#" | cut -f $i); \
                >             grep -w $id vignette/iraptus.csv; \
                >         done | cut -d "," -f 4,5 | sed s/","/"\t"/g > vignette/iraptus.locs 
		   
This filtering results in 1951 SNPs from 95 individuals. These values are included in our below ``disperseNN2`` preprocessing command.
This preprocessing step will take a while (maybe an hour), so it's a good time to get some coffee:

.. code-block:: console
		
		(.venv) $ disperseNN2 \
		>             --out vignette/output_dir \
		>	      --seed 12345 \
		>	      --preprocess \
		>	      --num_snps 1951 \
		>	      --n 95 \
		>	      --tree_list vignette/tree_list.txt \
		>	      --target_list vignette/target_list.txt \
		>	      --empirical vignette/iraptus \
		>	      --hold_out 100










   


		       


.. _vignette_training:

3. Training
-----------

In the below ``disperseNN2`` training command, there are two options that bear a bit of explanation.
In the example data we are working with there are 95 individuals, and so ${95\choose 2}$ = 4465 pairs of individuals.
We set ``--pairs`` to 1000 to reduce the number of pairwise comparisons used and thus the memory requirement.
Further our architecture only considers a subset of pairs on the backward pass for gradient computation in first half of the network;
this number is chosen with ``--pairs_encode``.
In our paper we found that using 100 for ``--pairs_encode`` reduced memory significantly without affecting accuracy;
100 is not a rule of thumb and you should try different values in your analysis.
Training on ~50 CPU cores, or one GPU, will take approximately 20 minutes.

.. code-block:: console

                (.venv) $ disperseNN2 \
		>             --out vignette/output_dir \
		> 	      --seed 12345 \
		> 	      --train \
		>             --max_epochs 100 \
		>             --validation_split 0.2 \
		>             --batch_size 10 \
		>             --learning_rate 1e-4 \
		>             --pairs 1000 \
		>             --pairs_encode 100 \
		>             --threads $num_threads \
		>	      > vignette/output_dir/training_history_12345.txt

After the run completes, you can visualize the training history. This will create a plot of the training and validation loss
declining over epochs of training:

.. ``vignette/output_dir/training_history_12345.txt_plot.pdf``:

.. code-block:: console

                (.venv) $ disperseNN2 --plot_history vignette/output_dir/training_history_12345.txt
		
.. figure:: training_vignette.png
   :scale: 50 %
   :alt: training_plot

   Plot of training history. X-axis the training iteration, and Y-axis is mean squared error.

This plot shows that the validation loss decreases over time, without too much under- or over-fitting.
		





		       






.. _vignette_validation:

4. Validation
-------------

Next, we will validate the trained model on simulated test data that was held out from training.

.. code-block:: console

                (.venv) $ disperseNN2 \
		>             --out vignette/output_dir \
                >             --seed 12345 \		
		>             --predict \
		>             --batch_size 10 \
		>             --num_pred 100 \
		>             --threads $num_threads


And some code for visualizing the predictions:

.. code-block:: console

		(.venv) $ pip install pandas
		(.venv) $ python -c 'import pandas as pd; from matplotlib import pyplot as plt; x = pd.read_csv("vignette/output_dir/Test/predictions_12345.txt", sep="\t", header=None); plt.scatter(x[0], x[1]); plt.xlabel("true"); plt.ylabel("predicted"); plt.savefig("results.pdf", format="pdf", bbox_inches="tight")'

		
Below is the plot of the predictions, ``results.pdf``:
		
.. figure:: results_vignette.png
   :scale: 25 %
   :alt: results_plot

   Validation results. True :math:`\sigma` is on the x-axis and predicted values are on the y-axis. The dashed line is :math:`x=y`.
		       
The predictions are reasonably close to the expected values, meaning there is some signal for dispersal rate. The training run was successful.

.. However, we are currently underestimating towards the larger end of the :math:`\sigma` range. This might be alleviated by using (i) a larger training set, (ii) more generatinos spatial, (iii) larger sample size, or (iv) or more SNPs.








.. _vignette_empirical:

5. Empirical application
------------------------

Since we are satisfied with the performance of the model on the held-out test set, we can finally predict σ in our empirical data.

.. code-block:: console

		(.venv) $ disperseNN2 \
		>             --out vignette/output_dir \
                >             --seed 12345 \		
		>	      --predict \
		>	      --empirical vignette/iraptus \
		>	      --batch_size 10 \
		>             --num_reps 10 \
                >     	      --threads	$num_threads

The final empirical results are stored here:

.. code-block:: console

		(.venv) $ cat vignette/output_dir/empirical_12345.txt
		vignette/iraptus rep0 2.0844352861
		vignette/iraptus rep1 2.5100254281
		vignette/iraptus rep2 2.2156516418
		vignette/iraptus rep3 2.4447924536
		vignette/iraptus rep4 2.505104107
		vignette/iraptus rep5 2.2632444932
		vignette/iraptus rep6 2.4132488538
		vignette/iraptus rep7 1.7120651222
		vignette/iraptus rep8 1.8796258058
		vignette/iraptus rep9 2.3444896444


		
**Interpretation**.
The output, :math:`\sigma`, is an estimate for the standard deviation of the Gaussian dispersal kernel from our training simulations. In addition, the same parameter was used for the mating distance (and competition distance). Therefore, to get the distance to a random parent, i.e., effective :math:`\sigma`,  we would apply a posthoc correction of :math:`\sqrt{\frac{3}{2}} \times \sigma` (see original disperseNN paper for details). In this example, we trained with only 100 generations spatial, hence the dispersal rate estimate reflects demography in the recent past.




.. _google_colab_notebook:

6. Google Colab notebook
------------------------

We have also setup a google colab notebook that runs through this example in a GPU enabled cloud setting.
We highly recommend checking out this notebook for the impatient, as we provide pre-processed simulation
results and a fully executable training/validation/prediction pipeline. The notebook can be found here:
`colab notebook <https://colab.research.google.com/github/kr-colab/disperseNN2/blob/main/docs/disperseNN2_vignette.ipynb>`_.



