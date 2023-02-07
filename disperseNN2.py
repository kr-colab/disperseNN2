
# e.g. python disperseNN2/disperseNN2.py --out temp1 --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --min_n 10 --max_n 10 --mu 1e-15 --seed 12345 --tree_list ../Maps/Boxes84/tree_list.txt --target_list ../Maps/Boxes84/target_list.txt --recapitate False --mutate True --phase 1 --polarize 2 --sampling_width 1 --num_samples 50 --edge_width 3 --train --learning_rate 1e-4 --grid_coarseness 50 --upsample 6 --num_pairs 45 --gpu_index any

import os
import argparse
import tskit
from sklearn.model_selection import train_test_split
from check_params import *
from read_input import *
from process_input import *
from data_generation import DataGenerator
import gpustat
import itertools

def load_dl_modules():
    print("loading bigger modules")
    import numpy as np
    global tf
    import tensorflow as tf
    from tensorflow import keras
    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train", action="store_true", default=False, help="run training pipeline"
)
parser.add_argument(
    "--predict", action="store_true", default=False, help="run prediction pipeline"
)
parser.add_argument(
    "--preprocess",
    action="store_true",
    default=False,
    help="create preprocessed tensors from tree sequences",
)
parser.add_argument(
    "--preprocessed",
    action="store_true",
    default=False,
    help="use preprocessed tensors, rather than tree sequences, as input",
)
parser.add_argument("--empirical", default=None,
                    help="prefix for vcf and locs")
parser.add_argument(
    "--target_list", help="list of filepaths to targets (sigma).", default=None)
parser.add_argument(
    "--tree_list", help="list of tree filepaths.", default=None)
parser.add_argument(
    "--edge_width",
    help="crop a fixed width from each edge of the map; enter 'sigma' to set edge_width equal to sigma ",
    default=None,
    type=str,
)
parser.add_argument(
    "--sampling_width", help="just the sampling area", default=None, type=float
)
parser.add_argument(
    "--num_snps",
    default=None,
    type=int,
    help="maximum number of SNPs across all datasets (for pre-allocating memory)",
)
parser.add_argument(
    "--num_pred", default=None, type=int, help="number of datasets to predict on"
)
parser.add_argument(
    "--min_n",
    default=None,
    type=int,
    help="minimum sample size",
)
parser.add_argument(
    "--max_n",
    default=None,
    type=int,
    help="maximum sample size",
)
parser.add_argument(
    "--mu",
    help="beginning mutation rate: mu is increased until num_snps is achieved",
    default=1e-15,
    type=float,
)
parser.add_argument("--rho", help="recombination rate",
                    default=1e-8, type=float)
parser.add_argument(
    "--num_samples",
    default=1,
    type=int,
    help="number of repeated samples (each of size n) from each tree sequence",
)
parser.add_argument(
    "--num_reps",
    default=1,
    type=int,
    help="number of replicate-draws from the genotype matrix of each sample",
)
parser.add_argument(
    "--validation_split",
    default=0.2,
    type=float,
    help="0-1, proportion of samples to use for validation.",
)
parser.add_argument("--batch_size", default=1, type=int, help="batch size for training")
parser.add_argument("--max_epochs", default=1000,
                    type=int, help="max epochs for training")
parser.add_argument(
    "--patience",
    type=int,
    default=100,
    help="n epochs to run the optimizer after last improvement in validation loss.",
)
parser.add_argument(
    "--dropout",
    default=0,
    type=float,
    help="proportion of weights to zero at the dropout layer.",
)
parser.add_argument(
    "--recapitate", type=str, help="recapitate on-the-fly; True or False"
)
parser.add_argument(
    "--mutate", type=str, help="add mutations on-the-fly; True or False"
)
parser.add_argument("--crop", default=None, type=float, help="map-crop size")
parser.add_argument(
    "--out", help="file name stem for output", default=None, required=True
)
parser.add_argument("--seed", default=None, type=int, help="random seed.")
parser.add_argument("--gpu_index", default="-1", type=str,
                    help="index of gpu. To avoid GPUs, skip this flag or say '-1'. To use any available GPU say 'any' ")
parser.add_argument(
    "--load_weights",
    default=None,
    type=str,
    help="Path to a _weights.hdf5 file to load weight from previous run.",
)
parser.add_argument(
    "--load_model",
    default=None,
    type=str,
    help="Path to a _model.hdf5 file to load model from previous run.",
)
parser.add_argument(
    "--phase",
    default=1,
    type=int,
    help="1 for unknown phase, 2 for known phase",
)
parser.add_argument(
    "--polarize",
    default=2,
    type=int,
    help="2 for major/minor, 1 for ancestral/derived",
)
parser.add_argument(
    "--keras_verbose",
    default=1,
    type=int,
    help="verbose argument passed to keras in model training. \
                    0 = silent. 1 = progress bars for minibatches. 2 = show epochs. \
                    Yes, 1 is more verbose than 2. Blame keras.",
)
parser.add_argument(
    "--threads",
    default=1,
    type=int,
    help="num threads.",
)
parser.add_argument("--samplewidth_list", help="", default=None)
parser.add_argument("--geno_list", help="", default=None)
parser.add_argument("--loc_list", help="", default=None)
parser.add_argument(
    "--training_params", help="params used in training: sigma mean and sd, max_n, num_snps", default=None
)
parser.add_argument(
    "--learning_rate",
    default=1e-3,
    type=float,
    help="learning rate.",
)
parser.add_argument("--combination_size", help="", default=2, type=int)
parser.add_argument("--grid_coarseness", help="", default=50, type=int)
parser.add_argument("--upsample", help="number of upsample layers", default=6, type=int)
parser.add_argument("--num_pairs", help="number of pairs to subsample", default=45, type=int)


args = parser.parse_args()
check_params(args)

#config = tf.ConfigProto(device_count={"CPU": 40})
#from keras import backend
#backend.tensorflow_backend.set_session(tf.Session(config=config))



def load_network():
    # set seed, gpu

    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
    if args.gpu_index != 'any':  # 'any' will search for any available GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    else:
        stats = gpustat.GPUStatCollection.new_query()
        ids = map(lambda gpu: int(gpu.entry['index']), stats)
        ratios = map(lambda gpu: float(
            gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # update conv+pool iterations based on number of SNPs
    num_conv_iterations = int(np.floor(np.log10(args.num_snps))-2)
    if num_conv_iterations < 0:
        num_conv_iterations = 0

    # cnn architecture
    conv_kernal_size = 2
    pooling_size = 10
    filter_size = 64
    combinations = list(itertools.combinations(range(args.max_n), args.combination_size))
    combinations = random.sample(combinations, args.num_pairs)

    # load inputs
    geno_input = tf.keras.layers.Input(shape=(args.num_snps, args.max_n)) 
    loc_input = tf.keras.layers.Input(shape=(2, args.max_n))

    # initialize shared layers
    CONV_1 = tf.keras.layers.Conv1D(filter_size, kernel_size=conv_kernal_size, activation="relu")
    CONV_2 = tf.keras.layers.Conv1D(filter_size +44, kernel_size=conv_kernal_size, activation="relu")
    DENSE_1 = tf.keras.layers.Dense(128, activation="relu")

    # convolutions for each pair
    first_pair = True     
    for comb in combinations:
        h = tf.gather(geno_input, comb, axis = 2)
        h = CONV_1(h)
        h = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)(h)
        h = CONV_2(h)        
        h = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)(h)            
        h = DENSE_1(h)                             
        h = tf.keras.layers.Flatten()(h)        
        l = tf.gather(loc_input, comb, axis = 2)
        l = tf.keras.layers.Flatten()(l)
        if first_pair == True: 
            h0 = h
            l0 = l
            first_pair = False
        else:
            h0 = tf.keras.layers.concatenate([h0,h])
            l0 = tf.keras.layers.concatenate([l0,l])

    # reshape conv. output and locs
    conv_output_shape = h0.shape # size of the concatenated convolution outputs
    single_conv_shape =int( conv_output_shape[1] / args.num_pairs) # size of convolution output per combination
    h0 = tf.keras.layers.Reshape((args.num_pairs,single_conv_shape), input_shape=conv_output_shape)(h0) 
    l0 = tf.keras.layers.Reshape((args.num_pairs,4), input_shape=conv_output_shape)(l0)

    # upsampling params
    minTensor = 10 # (theoretical min would be 5, because locs = 4 pieces of data)
    sizeOut = 500
    num_layers = int(args.upsample) # num dense layers to apply post-convolutions; includes the base map, and the output layer too, so at least 2.
    diff = np.log(sizeOut) - np.log(minTensor)
    bin_size = diff / (num_layers-1)   
    breaks = [0]                                                   
    for i in range(num_layers):                                  
        current_break = np.log(minTensor) + (bin_size * i)
        current_break = np.exp(current_break)                    
        current_break = np.round(current_break)                  
        breaks.append(int(current_break))                        
    early_breaks,late_breaks = [],[]
    for b in breaks:
        if b <= args.num_pairs:
            early_breaks.append(b)
        elif b > args.num_pairs:
            late_breaks.append(b)

    print(h0.shape)
    print(breaks)
    print(early_breaks,late_breaks)

    # further condensing genotype information via dense layers
    hs = [] 
    hs.append(tf.keras.layers.Dense(breaks[1]-4, activation='relu')(h0))
    for l in range(2, min(len(early_breaks),len(breaks)-1)): # (the min is for cases with zero late breaks)
        hs.append(tf.keras.layers.Dense(breaks[l]-4, activation='relu')(hs[l-2]))

    # early upsamples (condensing feature block)
    for l in range(1, min(len(early_breaks), len(breaks)-1)): # (the min is for cases with zero late breaks)
        print("\n early:",l)
        feature_block = tf.keras.layers.Dense(breaks[l]-4, activation='relu')(hs[l-1]) 
        print(feature_block.shape)
        feature_block =  tf.keras.layers.concatenate([feature_block,l0])                                                 
        print(feature_block.shape)
        feature_block = tf.keras.layers.Permute((2,1))(feature_block)                                                                      
        if l > 1: # except for first layer
            feature_block = tf.keras.layers.concatenate([h, feature_block]) # ~~~ skip connection engaged ~~~
        print(feature_block.shape)
        h = tf.keras.layers.Dense(breaks[l], activation='relu')(feature_block)
        print(h.shape)
        h = tf.keras.layers.Reshape((breaks[l],breaks[l],1), input_shape=(breaks[l],breaks[l]))(h)
        print(h.shape)
        h = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(breaks[l+1]-breaks[l]+1,breaks[l+1]-breaks[l]+1), activation="relu")(h)
        print(h.shape)
        h = tf.keras.layers.Reshape((breaks[l+1],breaks[l+1]), input_shape=(breaks[l+1],breaks[l+1],1))(h)
        print(h.shape)

    # freeze feature block (note: feature_block is also used below "output")
    print("\n freeze:", h0.shape, l0.shape)
    feature_block =  tf.keras.layers.concatenate([h0,l0])
    print(feature_block.shape)

    if late_breaks:
        # later upsamples (padding feature block)                                                                                                                                                    
        for l in range(len(early_breaks),len(breaks)-1):                                                                                       
            print("\n late:",l)
            paddings = tf.constant([[0,0], [0,breaks[l]-args.num_pairs], [0,0]]) # pad feature block out to the upsampled-size
            feature_block_padded = tf.pad(feature_block, paddings, "CONSTANT")
            print(feature_block_padded.shape)
            h = tf.keras.layers.concatenate([h, feature_block_padded]) # ~~~ skip connection engaged ~~~                        
            print(h.shape)
            h = tf.keras.layers.Dense(breaks[l], activation='relu')(h)                                                                         
            print(h.shape)                                                                                                                     
            h = tf.keras.layers.Reshape((breaks[l],breaks[l],1), input_shape=(breaks[l],breaks[l]))(h)                                         
            print(h.shape)                                                                                                                     
            h = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(breaks[l+1]-breaks[l]+1,breaks[l+1]-breaks[l]+1), activation="relu")(h)
            print(h.shape)                                                                                                                     
            h = tf.keras.layers.Reshape((breaks[l+1],breaks[l+1]), input_shape=(breaks[l+1],breaks[l+1],1))(h)                                 
            print(h.shape)                                                                                                                     

    # output
    if args.num_pairs > breaks[-1]: # condense feature block
        print("\n output, condensing:")
        feature_block = tf.keras.layers.Dense(breaks[-1]-4, activation='relu')(hs[-1])
        print(feature_block.shape)
        feature_block =  tf.keras.layers.concatenate([feature_block,l0])
        print(feature_block.shape)
        feature_block = tf.keras.layers.Permute((2,1))(feature_block)
        feature_block = tf.keras.layers.concatenate([h, feature_block]) # ~~~ skip connection engaged ~~~   
    elif args.num_pairs <= breaks[-1]: # pad feature block
        print("\n output, padding:")
        paddings = tf.constant([[0,0], [0,breaks[-1]-args.num_pairs], [0,0]]) # pad feature block out to the upsampled-size        
        feature_block_padded = tf.pad(feature_block, paddings, "CONSTANT")
        print(feature_block_padded.shape)
        h = tf.keras.layers.concatenate([h, feature_block_padded]) # ~~~ skip connection engaged ~~~
        print(h.shape)
    output = tf.keras.layers.Dense(sizeOut, activation='linear')(h) 
    print(output.shape, "\n")

    # model overview and hyperparams
    model = tf.keras.Model(
        inputs = [geno_input, loc_input],                                                                                                      
        outputs = [output],
    )
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model.compile(loss="mse", optimizer=opt)
    model.summary()
    print("total params:", np.sum([np.prod(v.shape) for v in model.trainable_variables]), "\n")

    # load weights
    if args.load_weights is not None:
        print("loading saved weights")
        model.load_weights(args.load_weights)
    else:
        if args.train == True and args.predict == True:
            weights = args.out + "/pwConv_" + str(args.seed) + "_model.hdf5"
            print("loading weights:", weights)
            model.load_weights(weights)
        elif args.predict == True:
            print("where is the saved model? (via --load_weights)")
            exit()

    # callbacks
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath= args.out + "/pwConv_" + str(args.seed) + "_model.hdf5",
        verbose=args.keras_verbose,
        save_best_only=True,
        saveweights_only=False,
        monitor="val_loss",
        period=1,
    )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=args.patience
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=int(args.patience/10),
        verbose=args.keras_verbose,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )

    return model, checkpointer, earlystop, reducelr


def make_generator_params_dict(
    targets, trees, shuffle, genos, locs, sample_widths
):
    params = {
        "targets": targets,
        "trees": trees,
        "num_snps": args.num_snps,
        "min_n": args.min_n,
        "max_n": args.max_n,
        "batch_size": args.batch_size,
        "mu": args.mu,
        "threads": args.threads,
        "shuffle": shuffle,
        "rho": args.rho,
        "baseseed": args.seed,
        "recapitate": args.recapitate,
        "mutate": args.mutate,
        "crop": args.crop,
        "sampling_width": args.sampling_width,
        "edge_width": args.edge_width,
        "phase": args.phase,
        "polarize": args.polarize,
        "sample_widths": sample_widths,
        "genos": genos,
        "locs": locs,
        "preprocessed": args.preprocessed,
        "num_reps": args.num_reps,
        "combination_size": args.combination_size,
        "grid_coarseness": args.grid_coarseness,
    }
    return params


def prep_trees_and_train():

    # tree sequences
    print(args.tree_list, flush=True)
    trees = read_dict(args.tree_list)
    total_sims = len(trees)

    # read targets                                                                 
    print("reading targets from tree sequences: this should take several minutes", flush=True)
    targets = []
    maps = read_dict(args.target_list)
    for i in range(total_sims):
        arr = read_map(maps[i], args.grid_coarseness)
        targets.append(arr)
        print("finished with " + str(i), flush=True)


    # normalize targets                                                               
    meanSig = np.mean(targets)
    sdSig = np.std(targets)
    np.save(f"{args.out}_training_params", [
            meanSig, sdSig, args.max_n, args.num_snps])
    targets = [(x - meanSig) / sdSig for x in targets]  # center and scale
    targets = dict_from_list(targets)    

    # split into val,train sets
    sim_ids = np.arange(0, total_sims)
    train, val = train_test_split(sim_ids, test_size=args.validation_split)
    if len(val)*args.num_samples % args.batch_size != 0 or len(train)*args.num_samples % args.batch_size != 0:
        print(
            "\n\ntrain and val sets each need to be divisible by batch_size; otherwise some batches will have missing data\n\n"
        )
        exit()

    # organize "partitions" to hand to data generator
    partition = {}
    partition["train"] = []
    partition["validation"] = []
    for i in train:
        for j in range(args.num_samples):
            partition["train"].append(i)
    for i in val:
        for j in range(args.num_samples):
            partition["validation"].append(i)

    # initialize generators
    params = make_generator_params_dict(
        targets=targets,
        trees=trees,
        shuffle=True,
        genos=None,
        locs=None,
        sample_widths=None,
    )
    training_generator = DataGenerator(partition["train"], **params)
    validation_generator = DataGenerator(partition["validation"], **params)

    # train
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()
    print("training!")
    history = model.fit(
        x=training_generator,
        use_multiprocessing=True,
        workers=args.threads,
        epochs=args.max_epochs,
        shuffle=False,  # (redundant with shuffling inside the generator)
        verbose=args.keras_verbose,
        validation_data=validation_generator,
        callbacks=[checkpointer, earlystop, reducelr],
    )

    return


def prep_preprocessed_and_train():

    # read targets
    print("loading data paths; this could take a while if the lists are very long")
    print("\ttargets")
    sys.stderr.flush()
    targets,genos,locs = dict_from_preprocessed(args.out)
    total_sims = len(targets)

    # split into val,train sets
    sim_ids = np.arange(0, total_sims)
    train, val = train_test_split(sim_ids, test_size=args.validation_split)
    if len(val)*args.num_samples % args.batch_size != 0 or len(train)*args.num_samples % args.batch_size != 0:
        print(
            "\n\ntrain and val sets each need to be divisible by batch_size; otherwise some batches will have missing data\n\n"
        )
        exit()

    # organize "partitions" to hand to data generator
    partition = {}
    partition["train"] = list(train)
    partition["validation"] = list(val)

    # initialize generators
    params = make_generator_params_dict(
        targets=targets,
        trees=None,
        shuffle=True,
        genos=genos,
        locs=locs,
        sample_widths=None,
    )
    training_generator = DataGenerator(partition["train"], **params)
    validation_generator = DataGenerator(partition["validation"], **params)

    # train
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()
    print("training!")
    history = model.fit_generator(
        generator=training_generator,
        use_multiprocessing=False,
        epochs=args.max_epochs,
        shuffle=False, # (redundant with shuffling inside generator)
        verbose=args.keras_verbose,
        validation_data=validation_generator,
        callbacks=[checkpointer, earlystop, reducelr],
    )

    return


def prep_empirical_and_pred():

    # grab mean and sd from training distribution
    meanSig, sdSig, args.max_n, args.num_snps = np.load(args.training_params)
    args.max_n = int(args.max_n)
    args.num_snps = int(args.num_snps)

    # project locs
    locs = read_locs(args.empirical + ".locs")
    locs = np.array(locs)
    sampling_width = project_locs(locs)
    print("sampling_width:", sampling_width)
    sampling_width = np.reshape(sampling_width, (1))

    # load model
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()

    # convert vcf to geno matrix
    for i in range(args.num_reps):
        test_genos = vcf2genos(
            args.empirical + ".vcf", args.max_n, args.num_snps, args.phase
        )
        ibd(test_genos, locs, args.phase, args.num_snps)
        test_genos = np.reshape(
            test_genos, (1, test_genos.shape[0], test_genos.shape[1])
        )
        dataset = args.empirical + "_" + str(i)
        prediction = model.predict([test_genos, sampling_width])
        unpack_predictions(prediction, meanSig, sdSig, None, None, dataset)

    return


def prep_preprocessed_and_pred():

    # grab mean and sd from training distribution
    meanSig, sdSig, args.max_n, args.num_snps = np.load(args.training_params)
    args.max_n = int(args.max_n)
    args.num_snps = int(args.num_snps)

    # load inputs
    targets = read_single_value(args.target_list)
    targets = np.log(targets)
    targets = dict_from_list(targets)
    genos = read_dict(args.geno_list)
    sample_widths = load_single_value_dict(args.samplewidth_list)
    # storing just the names, for output.
    file_names = read_dict(args.geno_list)
    total_sims = len(file_names)

    # organize "partition" to hand to data generator
    partition = {}
    if args.num_pred == None:
        args.num_pred = int(total_sims)
    simids = np.random.choice(np.arange(total_sims),
                              args.num_pred, replace=False)
    partition["prediction"] = simids

    # get generator ready
    params = make_generator_params_dict(
        targets=targets,
        trees=None,
        shuffle=False,
        genos=genos,
        locs=None,
        sample_widths=sample_widths,
    )
    generator = DataGenerator(partition["prediction"], **params)

    # predict
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()
    print("predicting")
    predictions = model.predict_generator(generator)
    unpack_predictions(predictions, meanSig, sdSig,
                       targets, simids, file_names)

    return


def prep_trees_and_pred():

    # grab mean and sd from training distribution
    meanSig, sdSig, args.max_n, args.num_snps = np.load(args.training_params)
    args.max_n = int(args.max_n)
    args.num_snps = int(args.num_snps)

    # tree sequences                                                              
    print(args.tree_list)
    trees = read_dict(args.tree_list)
    total_sims = len(trees)

    # read targets                                                                
    print("reading targets from tree sequences: this should take several minutes")
    targets = []
    maps = read_dict(args.target_list)
    for i in range(total_sims):
        arr = read_map(maps[i], args.grid_coarseness)
        targets.append(arr)

    # organize "partition" to hand to data generator
    partition = {}
    if args.num_pred == None:
        args.num_pred = int(total_sims)

    simids = np.random.choice(np.arange(total_sims),
                              args.num_pred, replace=False)
    partition["prediction"] = simids

    # get generator ready
    params = make_generator_params_dict(
        targets=[None]*total_sims,
        trees=trees,
        shuffle=False,
        genos=None,
        locs=None,
        sample_widths=None,
    )
    generator = DataGenerator(partition["prediction"], **params)

    # predict
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()
    print("predicting")
    predictions = model.predict_generator(generator)
    unpack_predictions(predictions, meanSig, sdSig, targets, simids, trees)

    return


def unpack_predictions(predictions, meanSig, sdSig, targets, simids, file_names):
    if args.empirical == None:
        with open(f"{args.out}_predictions.txt", "w") as out_f:
            raes = []
            for i in range(args.num_pred):
                for r in range(args.num_reps):
                    pred_index = r + (i*args.num_reps) # predicted in "normalized space"    
                    trueval = targets[simids[i]] # not normalized as read in
                    prediction = predictions[pred_index]
                    prediction = (prediction * sdSig) + meanSig # back transform to real space

                    outline = ""
                    outline += file_names[simids[i]]
                    outline += "\t"
                    outline += str(500)
                    outline += "\t"
                    outline += "\t".join(list(map(str,trueval.flatten())))
                    outline += "\t"
                    outline += "\t".join(list(map(str,prediction.flatten())))
                    print(outline, file=out_f)

     #   print("mean RAE:", np.mean(raes))
    else:
        with open(f"{args.out}_predictions.txt", "a") as out_f:
            prediction = predictions[0][0]
            prediction = (prediction * sdSig) + meanSig
            prediction = np.exp(prediction)
            prediction = np.round(prediction, 10)
            print(file_names, prediction, file=out_f)

    return


def preprocess_trees():
    trees = read_list(args.tree_list)
    maps = read_list(args.target_list)
    total_sims = len(trees)

    # loop through maps to get mean and sd          
    if not os.path.isfile(args.out+"/mean_sd.txt"):
        targets = []
        for i in range(total_sims):
            arr = read_map(maps[i], args.grid_coarseness)
            targets.append(arr)
        meanSig = np.mean(targets)
        sdSig = np.std(targets)
        os.makedirs(args.out, exist_ok=True)
        np.save(args.out+"/mean_sd.txt", [meanSig,sdSig])

    # preprocess
    os.makedirs(os.path.join(args.out,"Maps",str(args.seed)), exist_ok=True)
    os.makedirs(os.path.join(args.out,"Genos",str(args.seed)), exist_ok=True)
    os.makedirs(os.path.join(args.out,"Locs",str(args.seed)), exist_ok=True)
    for i in range(total_sims):
        target = read_map(maps[i], args.grid_coarseness) 
        target = (target - meanSig) / sdSig
        params = make_generator_params_dict(
            targets=None,
            trees=None,
            shuffle=None,
            genos=None,
            locs=None,
            sample_widths=None,
        )
        training_generator = DataGenerator([None], **params) 
        geno_mat, locs = training_generator.sample_ts(trees[i], args.seed) 

        # write results
        mapfile = os.path.join(args.out,"Maps",str(args.seed),str(i)+".target")
        genofile = os.path.join(args.out,"Genos",str(args.seed),str(i)+".genos")
        locfile = os.path.join(args.out,"Locs",str(args.seed),str(i)+".locs")
        np.save(mapfile, target)
        np.save(genofile, geno_mat)
        np.save(locfile, locs)

    return





### main ###

# pre-process
if args.preprocess == True:
    print("starting pre-processing pipeline")
    preprocess_trees()

# train
if args.train == True:
    print("starting training pipeline")
    if args.preprocessed == False:
        print("using tree sequences")
        prep_trees_and_train()
    else:
        print("using pre-processed tensors")
        prep_preprocessed_and_train()

# predict
if args.predict == True:
    print("starting prediction pipeline")
    if args.empirical == None:
        print("predicting on simulated data")
        if args.preprocessed == True:
            print("using pre-processed tensors")
            prep_preprocessed_and_pred()
        else:
            print("using tree sequences")
            prep_trees_and_pred()
    else:
        print("predicting on empirical data")
        prep_empirical_and_pred()
