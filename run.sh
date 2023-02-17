#!/bin/bash                                                          
#SBATCH --partition=kerngpu
#SBATCH --job-name=pwConv         ### Job Name
#SBATCH --output=Output/pwConv.out         ### File in which to store job output
#SBATCH --error=Output/pwConv.err          ### File in which to store job error messages
#SBATCH --time=30-00:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1              ### Number of nodes needed for the job
#SBATCH --account=kernlab       ### Account used for job submission 
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 2
#SBATCH --exclude=n244
#SBATCH --gpus=3g.20gb:1 # --gpus=2g.10gb:1 --gpus=3g.20gb:1   



module load miniconda
conda info
conda activate /home/chriscs/Software/miniconda3/envs/disperseNN
conda info
module list
nvidia-smi


# notes:
# - 50gb ram, and 2g.10gb:1 for 45 pairs. 3g.20gb:1 for 450 pairs. (once you go bigger, you run into GPU memory limits, and then RAM limits)
date=0217
box=84
id=5
u=6
n=100
pairs=450
grid=4







               # TRAIN                                                              

### regular ###
python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --train --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any --num_pred 1 > Boxes$box"_"n$n"_"preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_"$date

### grid-sample ###
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess_grid/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --train --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any --sample_grid $grid > Boxes$box"_"n$n"_"preprocess_grid/pwConv.$id.txt_upsample$u"_"pairs$pairs"_grid"$grid"_date"$date

### image segmentation ### 
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --train --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any $segment > Boxes$box"_"n$n"_"preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_"$segment"_"date"$date

### 2-channel ###
#python disperseNN2/disperseNN2_dev_twoChannel.py --out Boxes$box"_"n$n"_"preprocess/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --train --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any --num_pred 1 > Boxes$box"_"n$n"_"preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_twoChannel"$date






                # PRED 
### regular ###
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess/ --num_snps 5000 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --predict --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any --training_params Boxes$box"_"n$n"_"preprocess/mean_sd.npy --load_weights Boxes$box"_"n$n"_"preprocess/pwConv_$id"_"model.hdf5 > Boxes$box"_"n$n"_"preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_"$date"_"predict



                                                                               



`