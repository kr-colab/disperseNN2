
# e.g. python disperseNN2/disperseNN2.py --out temp1 --num_snps 5000 --max_epochs 1000 --validation_split 0.2 --batch_size 10 --threads 1 --n 10 --mu 1e-15 --seed 12345 --tree_list ../Maps/Boxes84/tree_list.txt --target_list ../Maps/Boxes84/target_list.txt --recapitate False --mutate True --phase 1 --polarize 2 --num_samples 50 --edge_width 3 --train --learning_rate 1e-4 --grid_coarseness 50 --upsample 6 --pairs 45 --gpu_index any

# notes:
#     - learning rate 1e-3 doesn't work at all, for one sigma. Neither does 5e-4. 1e-4 works.
#     - 4950 pairs ran for a while on 80-GB ram, with 100 encode 100 estimate.

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
import matplotlib.pyplot as plt
import PIL.Image as Image

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
parser.add_argument(
    "--segment",
    action="store_true",
    default=False,
    help="predict number of (sigma) classes, and which class for each pixel",
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
    "--sampling_width", help="width of sampling area (-1 for random sampling width)", default=1.0, type=float,
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
    "--n",
    default=None,
    type=int,
    help="sample size",
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
    "--recapitate", action="store_true", help="recapitate tree sequences", default=False,
)
parser.add_argument(
    "--skip_mutate", action="store_true",help="skip adding mutations", default=False,
)
parser.add_argument("--crop", default=None, type=float, help="map-crop size")
parser.add_argument(
    "--out", help="file name stem for output", default=None, required=True
)
parser.add_argument("--seed", default=None, type=int, help="random seed.")
parser.add_argument("--gpu", default="-1", type=str,
                    help="index of gpu. To avoid GPUs, skip this flag or say '-1'. To use any available GPU say 'any' ")
parser.add_argument('--plot_history', default=False, type=str,
                    help="plot training history? default: False")
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
    "--training_params", help="params used in training: sigma mean and sd, n, num_snps", default=None
)
parser.add_argument(
    "--learning_rate",
    default=1e-3,
    type=float,
    help="learning rate.",
)
parser.add_argument("--grid_coarseness", help="TO DO", default=50, type=int)
parser.add_argument("--sample_grid", help="coarseness of grid for grid-sampling", default=None, type=int)
parser.add_argument("--upsample", help="number of upsample layers", default=6, type=int)
parser.add_argument("--pairs", help="number of pairs to include in the feature block", type=int)
parser.add_argument("--pairs_encode", help="number of pairs (<= pairs_encode) to use for gradient in the first part of the network", type=int)
parser.add_argument("--pairs_estimate", help="average the feature block over 'pairs_encode' / 'pairs_set' sets of pairs", type=int)


args = parser.parse_args()
check_params(args)





def load_network():
    # set seed, gpu
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
    if args.gpu != 'any':  # 'any' will search for any available GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        stats = gpustat.GPUStatCollection.new_query()
        ids = map(lambda gpu: int(gpu.entry['index']), stats)
        ratios = map(lambda gpu: float(
            gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

    # update conv+pool iterations based on number of SNPs
    num_conv_iterations = int(np.floor(np.log10(args.num_snps))-1)
    if num_conv_iterations < 0:
        num_conv_iterations = 0
        
    # organize pairs of individuals
    combinations = list(itertools.combinations(range(args.n), 2))
    combinations = random.sample(combinations, args.pairs)
    combinations_encode = random.sample(combinations, args.pairs_encode)
    combinations = list2dict(combinations) # (using tuples as dict keys seems to work)
    combinations_encode = list2dict(combinations_encode)
    
    # load inputs
    geno_input = tf.keras.layers.Input(shape=(args.num_snps, args.n)) 
    loc_input = tf.keras.layers.Input(shape=(2, args.n))

    # initialize shared layers
    CONV_LAYERS = []
    conv_kernal_size = 2
    pooling_size = 10
    for i in range(num_conv_iterations):                                             
        filter_size = 20 + 44*(i+1)
        CONV_LAYERS.append(tf.keras.layers.Conv1D(filter_size, kernel_size=conv_kernal_size, activation="relu", name="CONV_"+str(i)))
    DENSE_0 = tf.keras.layers.Dense(filter_size, activation="relu", name="DENSE_0") # matching size of final conv layer
    DENSE_1 = tf.keras.layers.Dense(args.pairs_estimate, activation="relu", name="DENSE_1")

    # convolutions for each pair
    hs = []
    ds = []
    for comb in combinations:
        h = tf.gather(geno_input, comb, axis = 2)
        if comb in combinations_encode:                                                                                      
            for i in range(num_conv_iterations):                                             
                h = CONV_LAYERS[i](h)
                h = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)(h)            
            h = DENSE_0(h)
        else: # cut gradient tape on some pairs to save memory
            for i in range(num_conv_iterations):
                h = tf.stop_gradient(CONV_LAYERS[i](h))
                h = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)(h)
            h = tf.stop_gradient(DENSE_0(h))            
        h = tf.keras.layers.Flatten()(h)        
        hs.append(h)
        l = tf.gather(loc_input, comb, axis = 2)
        d = l[:,:,0] - l[:,:,1]
        d = tf.norm(d, ord='euclidean', axis=1)
        ds.append(d)

    # reshape conv. output and locs
    h = tf.stack(hs, axis=1)
    d = tf.stack(ds, axis=1)                          
    d = tf.keras.layers.Reshape(((args.pairs, 1)))(d) 
    feature_block = tf.keras.layers.concatenate([h,d])
    print("\nfeature block:", feature_block.shape)

    # loop through sets of 'pairs_set' pairs 
    num_partitions = int(np.ceil(args.pairs / float(args.pairs_estimate)))
    row = 0
    dense_stack = []
    for p in range(num_partitions):
        part = feature_block[:,row:row+args.pairs_estimate,:]
        row += args.pairs_estimate
        if part.shape[1] < args.pairs_estimate: # if map dimension isn't divisible by 'size_pair_estimate'
            paddings = tf.constant([[0,0], [0,args.pairs_estimate-part.shape[1]], [0,0]])
            part = tf.pad(part, paddings, "CONSTANT")
        h0 = DENSE_1(part)
        dense_stack.append(h0)
    h = tf.stack(dense_stack, axis=1)
    h = tf.keras.layers.AveragePooling2D(pool_size=(num_partitions,1))(h)
    
    # compress down to one sigma estimate                                                                          
    h = tf.keras.layers.Flatten()(h)
    output = tf.keras.layers.Dense(1, activation="linear")(h)

    # model overview and hyperparams
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model = tf.keras.Model(
        inputs = [geno_input, loc_input],                                                                                                      
        outputs = [output],
    )
    model.compile(loss="mse", optimizer=opt)
    #model.summary()
    for v in model.trainable_variables:
        print(v.name)
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
        "n": args.n,
        "batch_size": args.batch_size,
        "mu": args.mu,
        "threads": args.threads,
        "shuffle": shuffle,
        "rho": args.rho,
        "baseseed": args.seed,
        "recapitate": args.recapitate,
        "skip_mutate": args.skip_mutate,
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
        "grid_coarseness": args.grid_coarseness,
        "segment": args.segment,
        "sample_grid": args.sample_grid,
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
        with open(maps[i]) as infile:
            arr = float(infile.readline().strip())
        targets.append(np.log(arr))

    # normalize targets                                                               
#    targets = np.array(targets)
    meanSig = np.mean(targets)
    sdSig = np.std(targets)
    np.save(f"{args.out}_training_params", [
            meanSig, sdSig, args.n, args.num_snps])
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
#    gpus = tf.config.experimental.list_physical_devices('GPU')
 #   for gpu in gpus:
  #      tf.config.experimental.set_memory_growth(gpu, True)
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

    # history = model.fit_generator(
    #     generator=training_generator,
    #     use_multiprocessing=False,
    #     epochs=args.max_epochs,
    #     shuffle=False,  # (redundant with shuffling inside the generator) 
    #     verbose=args.keras_verbose,
    #     validation_data=validation_generator,
    #     callbacks=[checkpointer, earlystop, reducelr],
    # )

    return


def prep_preprocessed_and_train():

    # read targets
    print("reading input paths", flush=True)
    targets,genos,locs = dict_from_preprocessed(args.out, args.segment)
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




def prep_preprocessed_and_pred():

    # grab mean and sd from training distribution
    meanSig, sdSig = np.load(args.out + "/mean_sd.npy")

    # load inputs
    targets,genos,locs = dict_from_preprocessed(args.out, args.segment)
    total_sims = len(targets)

    # organize "partition" to hand to data generator
    partition = {}
    if args.num_pred == None:
        args.num_pred = int(total_sims)
    simids = np.random.choice(np.arange(total_sims),
                              args.num_pred, replace=False)

    # get generator ready
    params = make_generator_params_dict(
        targets=targets,
        trees=None,
        shuffle=False,
        genos=genos,
        locs=locs,
        sample_widths=None,
    )

    # predict
    print("predicting")
    os.makedirs(args.out + "/Test_" + str(args.seed), exist_ok=True)
    if os.path.isfile(args.out + "/Test_" + str(args.seed) + "/pwConv_" + str(args.seed) + "_predictions.txt"):
        print("pred output exists; overwriting...")
        os.remove(args.out + "/Test_" + str(args.seed) + "/pwConv_" + str(args.seed) + "_predictions.txt")
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()
    for b in range(int(np.ceil(args.num_pred/args.batch_size))): # loop to alleviate memory
        simids_batch = simids[b*args.batch_size:(b+1)*args.batch_size]
        partition["prediction"] = np.array(simids_batch)
        generator = DataGenerator(partition["prediction"], **params)
        predictions = model.predict_generator(generator)
        unpack_predictions(predictions, meanSig, sdSig,
                           targets, simids_batch, targets)

    return


# def prep_trees_and_pred(): # *** never tested ***

#     # grab mean and sd from training distribution
#     meanSig, sdSig = np.load(args.out + "mean_sd.npy")
#     #args.n = int(args.n) # * what does this do?
#     #args.num_snps = int(args.num_snps)

#     # tree sequences                                                              
#     print(args.tree_list, flush=True)
#     trees = read_dict(args.tree_list)
#     total_sims = len(trees)

#     # read targets                                                                
#     print("reading targets from tree sequences: this should take several minutes")
#     targets = []
#     maps = read_dict(args.target_list)
#     if args.segment == False:
#         for i in range(total_sims):
#             arr = read_map(maps[i], args.grid_coarseness, args.segment)
#             targets.append(arr)
#     else:
#         targets_class = []
#         for i in range(total_sims):
#             arr = read_map(maps[i], args.grid_coarseness, args.segment)
#             targets.append(arr[:,:,0])
#             targets_class.append(arr[:,:,1:5])
#             print("finished with " + str(i), flush=True)
#     targets = dict_from_list(targets) # (unindent)
#     if args.segment == True:
#         targets_class = dict_from_list(targets_class)
#         targets = [targets, targets_class]

#     # organize "partition" to hand to data generator
#     partition = {}
#     if args.num_pred == None:
#         args.num_pred = int(total_sims)

#     simids = np.random.choice(np.arange(total_sims),
#                               args.num_pred, replace=False)
#     partition["prediction"] = simids

#     # get generator ready
#     params = make_generator_params_dict(
#         targets=[None]*total_sims,
#         trees=trees,
#         shuffle=False,
#         genos=None,
#         locs=None,
#         sample_widths=None,
#     )
#     generator = DataGenerator(partition["prediction"], **params)

#     # predict
#     load_dl_modules()
#     model, checkpointer, earlystop, reducelr = load_network()
#     print("predicting")
#     predictions = model.predict_generator(generator)
#     unpack_predictions(predictions, meanSig, sdSig, targets, simids, trees)

#     return


def unpack_predictions(predictions, meanSig, sdSig, targets, simids, file_names): 

    if args.empirical == None:
        with open(args.out + "/Test_" + str(args.seed) + "/pwConv_" + str(args.seed) + "_predictions.txt", "a") as out_f:
            raes = []
            for i in range(len(predictions)):

                # process output and read targets
                trueval = np.load(targets[simids[i]]) # read in normalized
                trueval = (trueval * sdSig) + meanSig
                prediction = predictions[i][0] # (500x500) 
                prediction = (prediction * sdSig) + meanSig 
                outline = "\t".join(map(str,[trueval, prediction]))
                print(outline, file=out_f)

    return


def preprocess_trees():
    trees = read_list(args.tree_list)
    maps = read_list(args.target_list)
    total_sims = len(trees)

    # loop through maps to get mean and sd          
    if os.path.isfile(args.out+"/mean_sd.npy"):
        meanSig,sdSig = np.load(args.out+"/mean_sd.npy")
    else:
        targets = []
        for i in range(total_sims):
            with open(maps[i]) as infile:                
                arr = np.log(float(infile.readline().strip()))
            targets.append(arr)
        meanSig = np.mean(targets)
        sdSig = np.std(targets)
        os.makedirs(args.out, exist_ok=True)
        np.save(args.out+"/mean_sd", [meanSig,sdSig])

    # initialize generator and some things
    os.makedirs(os.path.join(args.out,"Maps",str(args.seed)), exist_ok=True)
    os.makedirs(os.path.join(args.out,"Genos",str(args.seed)), exist_ok=True)
    os.makedirs(os.path.join(args.out,"Locs",str(args.seed)), exist_ok=True)
    params = make_generator_params_dict(
        targets=None,
        trees=None,
        shuffle=None,
        genos=None,
        locs=None,
        sample_widths=None,
    )
    training_generator = DataGenerator([None], **params)

    # preprocess
    for i in range(total_sims):
        mapfile = os.path.join(args.out,"Maps",str(args.seed),str(i)+".target")
        genofile = os.path.join(args.out,"Genos",str(args.seed),str(i)+".genos")
        locfile = os.path.join(args.out,"Locs",str(args.seed),str(i)+".locs")
        if os.path.isfile(genofile+".npy") == False or os.path.isfile(locfile+".npy") == False:
            geno_mat, locs = training_generator.sample_ts(trees[i], args.seed) 
            np.save(genofile, geno_mat)
            np.save(locfile, locs)
        if os.path.isfile(genofile+".npy") == True and os.path.isfile(locfile+".npy") == True: # (only add map if inputs successful)
            if os.path.isfile(mapfile+".npy") == False:
                with open(maps[i]) as infile:
                    target = np.log(float(infile.readline().strip()))
                target = (target - meanSig) / sdSig
                np.save(mapfile, target)
        
    return



def plot_history():
    loss,val_loss = [],[]
    with open(args.plot_history) as infile:
        for line in infile:
            if "val_loss:" in line:
                endofline = line.strip().split(" loss:")[-1]
                loss.append(float(endofline.split()[0]))
                val_loss.append(float(endofline.split()[3]))
    epochs = np.arange(len(loss))
    fig = plt.figure(figsize=(4,1.5),dpi=200)
    plt.rcParams.update({'font.size': 7})
    ax1=fig.add_axes([0,0,0.4,1])
    ax1.plot(epochs, val_loss, color="blue", lw=0.5, label='val_loss')
    ax1.set_xlabel("Epoch")
    ax1.plot(epochs, loss, color="red", lw=0.5, label='loss')
    ax1.legend()
    fig.savefig(args.plot_history+"_plot.pdf",bbox_inches='tight')



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

# plot training history               
if args.plot_history:
    plot_history()

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

