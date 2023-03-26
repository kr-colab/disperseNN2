#!/bin/bash                                                          
#SBATCH --partition=preempt
#SBATCH --job-name=prepXX         ### Job Name
#SBATCH --output=Output/prepXX.out         ### File in which to store job output
#SBATCH --error=Output/prepXX.err          ### File in which to store job error messages
#SBATCH --time=1-00:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1              ### Number of nodes needed for the job
#SBATCH --account=kernlab       ### Account used for job submission 
#SBATCH --mem=25gb
#SBATCH --cpus-per-task 1
#SBATCH --requeue



box=Boxes34
#box=temp1
n=100
snps=100000
segment=""       
#segment="--segment"
grid=""
#grid="--sample_grid 4"
#trees=/home/chriscs/kernlab/Maps/$box/tree_list.txt
#trees=/home/chriscs/kernlab/Maps/$box/tree_list_105_106.txt
trees=/home/chriscs/kernlab/Maps/$box/tree_list_train.txt 
#trees=/home/chriscs/kernlab/Maps/$box/tree_list_test.txt                                                              
#targets=/home/chriscs/kernlab//Maps/$box/map_list.txt
targets=/home/chriscs/kernlab//Maps/$box/map_list_train.txt
#targets=/home/chriscs/kernlab//Maps/$box/target_list.txt
#targets=/home/chriscs/kernlab//Maps/$box/target_list_105_106.txt
#targets=/home/chriscs/kernlab/Maps/$box/target_list_test.txt




# make individual jobs scripts
#     for i in {1..50}; do cat disperseNN2/preprocess.sh | sed s/XX/$i/g > Jobs/job$i.sh; done
#     for i in {1..50}; do sbatch Jobs/job$i.sh; done



module load miniconda
conda activate /home/chriscs/Software/miniconda3/envs/disperseNN




# regular example
#python disperseNN2/disperseNN2.py --out $box"_"n$n"_"preprocess --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 10 --n $n --mu 1e-15 --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --preprocess --learning_rate 1e-4 --grid_coarseness 50 --seed XX --tree_list $trees --target_list $targets

# grid sampling example
#python disperseNN2/disperseNN2.py --out $box"_"n$n"_"preprocess_grid --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 10 --n $n --mu 1e-15 --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --preprocess --learning_rate 1e-4 --grid_coarseness 50 --seed XX --tree_list $trees --target_list $targets $grid

# segment example- ordinal maps 
#python disperseNN2/disperseNN2.py --out $box"_"preprocess/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 1 --threads 1 --n 10 --mu 1e-15 --seed XX --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --preprocess --learning_rate 1e-4 --grid_coarseness 50 --upsample 6 --pairs 45 --segment --target_list $targets

# 2-channel
#python disperseNN2/disperseNN2_dev_twoChannel.py --out $box"_"n$n"_"preprocess --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 10 --n $n --mu 1e-15 --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --preprocess --learning_rate 1e-4 --grid_coarseness 50 --seed XX --tree_list $trees --target_list $targets

# one-sigma
python disperseNN2/disperseNN2_dev_oneSigma.py --out $box"_"n$n"_"$snps"snps_"preprocess_ONESIG --num_snps $snps --max_epochs 1000 --validation_split 0.2 --batch_size 1 --threads 1 --n $n --mu 1e-15 --seed XX --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --learning_rate 1e-4 --tree_list $trees --target_list $targets --preprocess



