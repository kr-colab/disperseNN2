#!/bin/bash                                                          
#SBATCH --partition=kerngpu
#SBATCH --job-name=train         ### Job Name
#SBATCH --output=Output/train.out         ### File in which to store job output
#SBATCH --error=Output/train.err          ### File in which to store job error messages
#SBATCH --time=30-00:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1              ### Number of nodes needed for the job
#SBATCH --account=kernlab       ### Account used for job submission 
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 2
#SBATCH --gpus=2g.10gb:1 # --gpus=2g.10gb:1 --gpus=3g.20gb:1




module load miniconda
conda info
conda activate /home/chriscs/Software/miniconda3/envs/disperseNN
conda info
module list
nvidia-smi


# TRAIN
# notes:
# - 50gb ram, and 2g.10gb:1 should work with current archtecture
id=5
u=6
n=100
pairs=4950
date=0207
#python disperseNN2/disperseNN2.py --out Boxes84_preprocess/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --train --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any > Boxes84_preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_"$date
#
python disperseNN2/disperseNN2.py --out Boxes84_n100_preprocess/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --train --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any  > Boxes84_n100_preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs

# # pred
# id=1
# v=31
# u=4
# python /home/chriscs/kernlab/Maps/Maps/Gaussian_sigma/disperseNN_v$v.py --out Boxes84/pwConv.$id.v$v --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 10 --min_n 10 --max_n 10 --mu 1e-15 --seed 12345 --tree_list Boxes84/tree_list.txt --target_list Boxes84/target_list.txt --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --predict --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --gpu_index any --training_params Boxes84/pwConv.$id.v$v"_"training_params.npy --load_weights Boxes84/pwConv.$id.v$v"_"model.hdf5 #--num_pred 1


                                                                               




# # ordinal
# id=2
# v=40
# u=4
# python /home/chriscs/kernlab/Maps/Maps/Gaussian_sigma/disperseNN_v$v.py --out Boxes84/pwConv.$id.v$v --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 10 --min_n 10 --max_n 10 --mu 1e-15 --seed 12345 --tree_list Boxes84/tree_list.txt --target_list Boxes84/target_list_ordinal.txt --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --train --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --gpu_index any  > Boxes84/pwConv.$id.txt_v$v"_"upsample$u

# # pred
# id=2
# v=40
# u=4
# python /home/chriscs/kernlab/Maps/Maps/Gaussian_sigma/disperseNN_v$v.py --out Boxes84/pwConv.$id.v$v --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 10 --min_n 10 --max_n 10 --mu 1e-15 --seed 12345 --tree_list Boxes84/tree_list.txt --target_list Boxes84/target_list_ordinal.txt --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --predict --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --gpu_index any --training_params Boxes84/pwConv.$id.v$v"_"training_params.npy --load_weights Boxes84/pwConv.$id.v$v"_"model.hdf5 --num_pred 1000

