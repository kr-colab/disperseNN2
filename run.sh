#!/bin/bash                                                          
#SBATCH --partition=kern
#SBATCH --job-name=pwConv         ### Job Name
#SBATCH --output=Output/pwConv.out         ### File in which to store job output
#SBATCH --error=Output/pwConv.err          ### File in which to store job error messages
#SBATCH --time=30-00:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1              ### Number of nodes needed for the job
#SBATCH --account=kernlab       ### Account used for job submission 
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 1
##SBATCH --exclude=n244,n273
##SBATCH --gpus=2g.10gb:1 
##SBATCH --gpus=3g.20gb:1



module load miniconda
conda activate /home/chriscs/Software/miniconda3/envs/disperseNN


# notes:
# - for maps: 50gb ram, and 2g.10gb:1 for 45 pairs. 3g.20gb:1 for 450 pairs. (once you go bigger, you run into GPU memory limits, and then RAM limits)
date=0313
box=105_106
id=1
u=6
n=23
pairs=100
grid=4
num_pred=10
DATE=$(date | awk '{print $2,$3}' | sed s/" "//g)




               # TRAIN                                                              
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess_ONESIG --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --n $n --mu 1e-15 --seed $id --recapitate False --mutate True --num_samples 50 --train --learning_rate 1e-4 --preprocessed --pairs $pairs --gpu_index -1 > Boxes$box"_"n$n"_"preprocess_ONESIG/out_one_sig.$id.txt_n$n"_"$pairs"pair_"$DATE







                # PRED 
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess_ONESIG --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 1 --threads 1 --n $n --mu 1e-15 --seed $id --recapitate False --mutate True --num_samples 50 --predict --learning_rate 1e-4 --preprocessed --pairs $pairs --load_weights Boxes105_106_n23_preprocess_ONESIG/out140_boxes105_noProj_model.hdf5 --num_pred 100 --gpu_index -1
#
python disperseNN2/disperseNN2.py --out out_one_sig --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --n 100 --mu 1e-15 --seed 38 --recapitate False --mutate True --learning_rate 1e-4 --preprocessed --pairs_encode 45 --pairs_downsample 45 --pairs_set 45 --predict --num_pred 1000 --load_weights out_one_sig/pwConv_38_model.hdf5 --gpu_index 1  
#
python disperseNN2/disperseNN2.py --out out_one_sig --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --n 100 --mu 1e-15 --seed 39 --recapitate False --mutate True --learning_rate 1e-4 --preprocessed --pairs_encode 4000 --pairs_downsample 100 --pairs_set 100 --predict --num_pred 1000 --load_weights out_one_sig/pwConv_39_model.hdf5 --gpu_index -1 





