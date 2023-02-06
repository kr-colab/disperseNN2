#!/bin/bash                                                          
#SBATCH --partition=preempt
#SBATCH --job-name=jobXX         ### Job Name
#SBATCH --output=Output/jobXX.out         ### File in which to store job output
#SBATCH --error=Output/jobXX.err          ### File in which to store job error messages
#SBATCH --time=7-00:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1              ### Number of nodes needed for the job
#SBATCH --account=kernlab       ### Account used for job submission 
#SBATCH --mem=50gb   # adjust for loading in all maps at once to calc. mean and sd.
#SBATCH --cpus-per-task 1
#SBATCH --requeue



box=Boxes87
n=10
trees=/home/chriscs/kernlab/Maps/$box/tree_list.txt
targets=/home/chriscs/kernlab//Maps/$box/target_list.txt



# make individual jobs scripts
#     for i in {1..50}; do cat disperseNN2/preprocess.sh | sed s/XX/$i/g > Jobs/job$i.sh; done
#     for i in {1..50}; do sbatch Jobs/job$i.sh; done



# preprocess
module load miniconda
conda activate /home/chriscs/Software/miniconda3/envs/disperseNN

python disperseNN2/disperseNN2.py --out $box"_"n$n"_"preprocess --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 10 --min_n $n --max_n $n --mu 1e-15 --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --preprocess --learning_rate 1e-4 --grid_coarseness 50 --seed XX --tree_list $trees --target_list $targets


