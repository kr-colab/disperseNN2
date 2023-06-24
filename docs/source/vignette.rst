Vignette: example workflow
==========================


This vignette shows a typical ``disperseNN2`` workflow. While the :doc:`usage` section was more brief, here we go into more detail for each step of the workflow:

:ref:`vignette_simulation`

:ref:`vignette_preprocessing`

:ref:`vignette_training`

:ref:`vignette_validation`

:ref:`vignette_empirical`

     

.. _vignette_simulation:

1. Simulation
-------------

mkdir temp_wd





mkdir temp_wd/TreeSeqs
for i in {1..100}
do
    sigma=$(python -c 'from scipy.stats import loguniform; print(loguniform.rvs(0.2,1.5))')
    echo "slim -d SEED=$i -d sigma=$sigma -d K=6 -d mu=0 -d r=1e-8 -d W=50 -d G=1e8 -d maxgens=100 -d OUTNAME=\"'temp_wd/TreeSeqs/output'\" SLiM_recipes/bat20.slim" >> temp_wd/sim_commands.txt
done
parallel -j 2 < temp_wd/sim_commands.txt






for i in {1..100};
do
    echo "python -c 'import tskit,msprime; ts=tskit.load(\"temp_wd/TreeSeqs/output_$i.trees\"); Ne=len(ts.individuals()); demography = msprime.Demography.from_tree_sequence(ts); demography[1].initial_size = Ne; ts = msprime.sim_ancestry(initial_state=ts, recombination_rate=1e-8, demography=demography, start_time=ts.metadata[\"SLiM\"][\"cycle\"],random_seed=$i,); ts.dump(\"temp_wd/TreeSeqs/output_$i"_"recap.trees\")'" >> temp_wd/recap_commands.txt
    echo temp_wd/TreeSeqs/output_$i"_"recap.trees >> temp_wd/tree_list.txt
done   
parallel -j 2 < temp_wd/recap_commands.txt







.. _vignette_preprocessing:

2. Preprocessing
----------------






.. _vignette_training:

3. Training
-----------










.. _vignette_validation:

4. Validation
-------------











.. _vignette_empirical:

5. Empirical application
------------------------

