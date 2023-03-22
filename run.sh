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

### regular ###
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --train --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any > Boxes$box"_"n$n"_"preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_"$DATE

### grid-sample ###
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess_grid/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --train --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any > Boxes$box"_"n$n"_"preprocess_grid/pwConv.$id.txt_upsample$u"_"pairs$pairs"_grid"$grid"_date"$DATE

### image segmentation ### 
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --train --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any $segment > Boxes$box"_"n$n"_"preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_"$segment"_"date"$DATE

### 2-channel ###
#python disperseNN2/disperseNN2_dev_twoChannel.py --out Boxes$box"_"n$n"_"preprocess/ --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --train --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any > Boxes$box"_"n$n"_"preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_twoChannel"$DATE

### one sigma ###
#python disperseNN2/disperseNN2_dev_oneSigma.py --out Boxes$box"_"n$n"_"preprocess_ONESIG --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --num_samples 50 --train --learning_rate 1e-4 --preprocessed --pairs $pairs --gpu_index -1 > Boxes$box"_"n$n"_"preprocess_ONESIG/out_one_sig.$id.txt_n$n"_"$pairs"pair_"$DATE







                # PRED 
### regular ###
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess/ --num_snps 5000 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --predict --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any --load_weights Boxes$box"_"n$n"_"preprocess/pwConv_$id"_"model.hdf5 --num_pred $num_pred > Boxes$box"_"n$n"_"preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_"$DATE"_"predict

### grid sample ###
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess_grid/ --num_snps 5000 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --predict --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any --load_weights Boxes$box"_"n$n"_"preprocess/pwConv_$id"_"model.hdf5 --num_pred $num_pred > Boxes$box"_"n$n"_"preprocess_grid/pwConv.$id.txt_upsample$u"_"pairs$pairs"_grid"$grid"_date"$DATE"_"predict

### image_segmentation ###
#python disperseNN2/disperseNN2.py --out Boxes$box"_"n$n"_"preprocess/ --num_snps 5000 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --predict --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --gpu_index any --load_weights Boxes$box"_"n$n"_"preprocess/pwConv_$id"_"model.hdf5 --num_pred $num_pred --segment > Boxes$box"_"n$n"_"preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_"$DATE"_"predict

### 2-channel ###
#python disperseNN2/disperseNN2_dev_twoChannel.py --out Boxes$box"_"n$n"_"preprocess/ --num_snps 5000 --batch_size 10 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --sampling_width 1 --num_samples 50 --edge_width 3 --predict --preprocessed --learning_rate 1e-4 --grid_coarseness 50 --upsample $u --pairs $pairs --load_weights Boxes$box"_"n$n"_"preprocess/pwConv_$id"_"model.hdf5 --num_pred $num_pred > Boxes$box"_"n$n"_"preprocess/pwConv.$id.txt_upsample$u"_"pairs$pairs"_"$DATE"_"predict

### one sigma ###                                                                          
# python disperseNN2/disperseNN2_dev_oneSigma.py --out Boxes$box"_"n$n"_"preprocess_ONESIG --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 1 --threads 1 --min_n $n --max_n $n --mu 1e-15 --seed $id --recapitate False --mutate True --num_samples 50 --predict --learning_rate 1e-4 --preprocessed --pairs $pairs --load_weights Boxes105_106_n23_preprocess_ONESIG/out140_boxes105_noProj_model.hdf5 --num_pred 100 --gpu_index -1
# train 105, test 106




