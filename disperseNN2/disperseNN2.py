#!/usr/bin/env python

# main code for disperseNN2

import os
import sys
import random
import argparse
from sklearn.model_selection import train_test_split
from disperseNN2.check_params import check_params
from disperseNN2.read_input import list2dict
from disperseNN2.read_input import read_list
from disperseNN2.read_input import read_locs
from disperseNN2.read_input import dict_from_preprocessed
from disperseNN2.process_input import project_locs, vcf2genos
import gpustat
import itertools
import numpy as np
import matplotlib.pyplot as plt


def load_dl_modules():  # load TF only if reading input is successfull
    print("loading bigger modules")
    global tf
    import tensorflow as tf
    global DataGenerator
    from disperseNN2.data_generation import DataGenerator  # (loads TF)

    return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--preprocess",
    action="store_true",
    default=False,
    help="create preprocessed tensors from tree sequences",
)
parser.add_argument(
    "--train", action="store_true", default=False, help="run training pipeline"
)
parser.add_argument(
    "--predict",
    action="store_true",
    default=False,
    help="run prediction pipeline"
)
parser.add_argument(
    "--empirical",
    default=None,
    help="prefix for vcf and locs"
)
parser.add_argument(
    "--target_list", help="list of filepaths to targets (sigma).", default=None
)
parser.add_argument(
    "--tree_list",
    help="list of tree filepaths.",
    default=None
)
parser.add_argument(
    "--edge_width",
    help="crop a fixed width from each edge of the map; \
    enter 'sigma' to set edge_width equal to sigma",
    default="0",
    type=str,
)
parser.add_argument(
    "--num_snps",
    default=None,
    type=int,
    help="number of SNPs",
)
parser.add_argument(
    "--num_pred",
    default=None,
    type=int,
    help="number of datasets to predict on"
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
parser.add_argument(
    "--rho",
    help="recombination rate",
    default=1e-8,
    type=float)
parser.add_argument(
    "--w",
    default=None,
    type=int,
    help="map width (will try to parse automatically if \"W\" provided to SLiM)",
)
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
    "--hold_out",
    default=0,
    type=int,
    help="integer, the number of tree sequences to hold out for testing.",
)
parser.add_argument(
    "--validation_split",
    default=0.2,
    type=float,
    help="proportion of training set (after holding out test data) \
    to use for validation during training.",
)
parser.add_argument(
    "--batch_size",
    default=10,
    type=int,
    help="batch size for training")
parser.add_argument(
    "--max_epochs", default=1000, type=int, help="max epochs for training"
)
parser.add_argument(
    "--patience",
    type=int,
    default=100,
    help="n epochs to run the optimizer after last improvement in \
    validation loss.",
)
parser.add_argument(
    "--dropout",
    default=0,
    type=float,
    help="proportion of weights to zero at the dropout layer.",
)
parser.add_argument(
    "--recapitate",
    action="store_true",
    help="recapitate tree sequences",
    default=False,
)
parser.add_argument(
    "--skip_mutate",
    action="store_true",
    help="skip adding mutations",
    default=False,
)
parser.add_argument(
    "--out",
    help="file name stem for output",
    default=None,
)
parser.add_argument("--seed", default=None, type=int, help="random seed.")
parser.add_argument(
    "--gpu",
    default="-1",
    type=str,
    help="gpu index (uses a single GPU). To avoid GPUs, \
    skip this flag or say '-1'. \
    To use any available GPU say 'any' ",
)
parser.add_argument(
    "--plot_history",
    default=False,
    type=str,
    help="plot training history? default: False",
)
parser.add_argument(
    "--load_weights",
    default=None,
    type=str,
    help="Path to a _weights.hdf5 file to load weight from previous run.",
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
    0 = silent. 1 = progress bars for minibatches. \
    2 = show epochs. \
    Yes, 1 is more verbose than 2. Blame keras.",
)
parser.add_argument(
    "--threads",
    default=1,
    type=int,
    help="num threads.",
)
parser.add_argument(
    "--training_mean_sd", help="sigma mean and sd from training", default=None
)
parser.add_argument(
    "--learning_rate",
    default=1e-4,
    type=float,
    help="learning rate.",
)
parser.add_argument("--grid_coarseness", help="TO DO", default=50, type=int)
parser.add_argument(
    "--sample_grid",
    help="coarseness of grid for grid-sampling",
    default=None,
    type=int
)
parser.add_argument(
    "--upsample",
    help="number of upsample layers",
    default=6,
    type=int)
parser.add_argument(
    "--pairs",
    help="number of pairs to include in the feature block",
    type=int,
    default=None,
)
parser.add_argument(
    "--pairs_encode",
    help="number of pairs (<= pairs_encode) to use for gradient \
    in the first part of the network",
    type=int,
    default=None,
)
parser.add_argument(
    "--force",
    action="store_true",
    default=False,
    help="force overwrite of existing data: \
    including existing model, or predictions"
)
parser.add_argument(
    "--no_shuffle_individuals",
    action="store_false",
    help="skip shuffling individuals within each dataset to get different pairs each batch and augment the training set (default is to shuffle)",
    default=True,
)
args = parser.parse_args()
check_params(args)
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)


def load_network():
    # set seed, gpu
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
    if args.gpu != "any":  # 'any' will search for any available GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        stats = gpustat.GPUStatCollection.new_query()
        ids = map(lambda gpu: int(gpu.entry["index"]), stats)
        ratios = map(
            lambda gpu: float(gpu.entry["memory.used"])
            / float(gpu.entry["memory.total"]),
            stats,
        )
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(bestGPU)
    if args.threads > 1:
        args.threads = int(
            np.floor(args.threads / 2)
        )  # in practice it uses double the specified threads
    tf.config.threading.set_intra_op_parallelism_threads(
        args.threads
    )  # limits (and sets) threads used during training
    tf.config.threading.set_inter_op_parallelism_threads(
        args.threads
    )  # this one is needed too.

    # allocate gpu memory dynamically
    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpu_devices:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    # update conv+pool iterations based on number of SNPs
    num_conv_iterations = int(np.floor(np.log10(args.num_snps)) - 1)
    if num_conv_iterations < 0:
        num_conv_iterations = 0

    # organize pairs of individuals
    combinations = list(itertools.combinations(range(args.n), 2))
    combinations = random.sample(combinations, args.pairs)
    combinations_encode = random.sample(combinations, args.pairs_encode)
    combinations = list2dict(combinations)
    combinations_encode = list2dict(combinations_encode)

    # load inputs
    geno_input = tf.keras.layers.Input(shape=(args.num_snps, args.n))
    loc_input = tf.keras.layers.Input(shape=(2, args.n))

    # initialize shared layers
    CONV_LAYERS = []
    conv_kernal_size = 2
    pooling_size = 10
    for i in range(num_conv_iterations):
        filter_size = 20 + 44 * (i + 1)
        CONV_LAYERS.append(
            tf.keras.layers.Conv1D(
                filter_size,
                kernel_size=conv_kernal_size,
                activation="relu",
                name="CONV_" + str(i),
            )
        )
    DENSE_0 = tf.keras.layers.Dense(128, activation="relu", name="DENSE_0")

    # convolutions for each pair
    hs = []
    for comb in combinations:
        h = tf.gather(geno_input, comb, axis=2)
        locs = tf.gather(loc_input, comb, axis=2)
        d = locs[:, :, 0] - locs[:, :, 1]
        d = tf.norm(d, ord="euclidean", axis=1)
        d = tf.keras.layers.Reshape((1,))(d)
        if comb in combinations_encode:
            for i in range(num_conv_iterations):
                h = CONV_LAYERS[i](h)
                h = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)(h)
            h = tf.keras.layers.Flatten()(h)
            h = tf.keras.layers.concatenate([h, d])
            h = DENSE_0(h)
        else:  # cut gradient tape on some pairs to save memory
            for i in range(num_conv_iterations):
                h = tf.stop_gradient(CONV_LAYERS[i](h))
                h = tf.keras.layers.AveragePooling1D(pool_size=pooling_size)(h)
            h = tf.keras.layers.Flatten()(h)
            h = tf.keras.layers.concatenate([h, d])
            h = tf.stop_gradient(DENSE_0(h))
        # (unindent)
        hs.append(h)
    # (unindent)
    feature_block = tf.stack(hs, axis=1)

    # flatten and final dense
    feature_block = tf.keras.layers.Flatten()(feature_block)
    output = tf.keras.layers.Dense(1,
                                   activation="linear",
                                   name="DENSE_1")(feature_block)

    # model overview and hyperparams
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    model = tf.keras.Model(
        inputs=[geno_input, loc_input],
        outputs=[output],
    )
    model.compile(loss="mse", optimizer=opt)
    # model.summary()
    for v in model.trainable_variables:
        print(v.name)
    print(
        "total params:",
        np.sum([np.prod(v.shape) for v in model.trainable_variables]),
        "\n",
    )

    # load weights
    if args.predict is True:
        if args.load_weights is None:
            weights = args.out + "/Train/disperseNN2_" + \
                str(args.seed) + "_model.hdf5"
        else:
            weights = args.load_weights
        print("loading weights:", weights)
        model.load_weights(weights)

    # callbacks
    checkpointer = tf.keras.callbacks.ModelCheckpoint(
        filepath=args.out + "/Train/disperseNN2_"
        + str(args.seed) + "_model.hdf5",
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
        patience=int(args.patience / 10),
        verbose=args.keras_verbose,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )

    return model, checkpointer, earlystop, reducelr


def make_generator_params_dict(
    targets,
    trees,
    shuffle_datasets,
    genos,
    locs,
    empirical_locs,
):
    params = {
        "targets": targets,
        "trees": trees,
        "num_snps": args.num_snps,
        "n": args.n,
        "batch_size": args.batch_size,
        "mu": args.mu,
        "shuffle_datasets": shuffle_datasets,
        "shuffle_individuals": args.no_shuffle_individuals,
        "rho": args.rho,
        "w": args.w,
        "baseseed": args.seed,
        "recapitate": args.recapitate,
        "skip_mutate": args.skip_mutate,
        "edge_width": args.edge_width,
        "phase": args.phase,
        "polarize": args.polarize,
        "genos": genos,
        "locs": locs,
        "grid_coarseness": args.grid_coarseness,
        "sample_grid": args.sample_grid,
        "empirical_locs": empirical_locs,
    }
    return params


def preprocess():
    # read lists
    trees = read_list(args.tree_list)
    target_paths = read_list(args.target_list)
    total_sims = len(trees)

    # separate training and test data
    train, test = train_test_split(np.arange(total_sims),
                                   test_size=args.hold_out)

    # loop through training targets to get mean and sd
    if os.path.isfile(args.out + "/preprocess_params.npy"):
        n, num_snps, meanSig, sdSig = np.load(
            args.out + "/preprocess_params.npy")
    else:
        targets = []
        for i in train:
            with open(target_paths[i]) as infile:
                arr = np.log(float(infile.readline().strip()))
            targets.append(arr)
        meanSig = np.mean(targets)
        sdSig = np.std(targets)
        os.makedirs(args.out + "/Train", exist_ok=True)
        np.save(
            args.out + "/preprocess_params",
            [args.n, args.num_snps, meanSig, sdSig]
        )

    # make directories
    os.makedirs(os.path.join(args.out,
                             "Train/Targets",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.out,
                             "Train/Genos",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.out,
                             "Train/Locs",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.out,
                             "Test/Targets",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.out,
                             "Test/Genos",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.out,
                             "Test/Locs",
                             str(args.seed)),
                exist_ok=True)

    # process
    load_dl_modules()  # (to get data generator)
    for i in range(total_sims):
        if i in test:
            split = "Test"
        else:
            split = "Train"
        targetfile = os.path.join(
            args.out, split, "Targets", str(args.seed), str(i) + ".target"
        )
        genofile = os.path.join(
            args.out, split, "Genos", str(args.seed), str(i) + ".genos"
        )
        locfile = os.path.join(
            args.out, split, "Locs", str(args.seed), str(i) + ".locs"
        )
        if (
            os.path.isfile(genofile + ".npy") is False
            or os.path.isfile(locfile + ".npy") is False
        ):
            if args.empirical is not None:
                locs = read_locs(args.empirical + ".locs")
                if len(locs) != args.n:
                    print("length of locs file doesn't match n")
                    exit()
                locs = project_locs(locs, trees[i])
            else:
                locs = None
            params = make_generator_params_dict(
                targets=None,
                trees=None,
                shuffle_datasets=None,
                genos=None,
                locs=None,
                empirical_locs=locs,
            )
            training_generator = DataGenerator([None], **params)
            geno_mat, locs = training_generator.sample_ts(trees[i], args.seed)
            np.save(genofile, geno_mat)
            np.save(locfile, locs)
        if (
            os.path.isfile(genofile + ".npy") is True
            and os.path.isfile(locfile + ".npy") is True
        ):  # (only add target if inputs successful)
            if os.path.isfile(targetfile + ".npy") is False:
                with open(target_paths[i]) as infile:
                    target = np.log(float(infile.readline().strip()))
                target = (target - meanSig) / sdSig
                np.save(targetfile, target)

    return


def train():
    # read targets
    print("reading input paths", flush=True)
    targets, genos, locs = dict_from_preprocessed(args.out + "/Train/")
    total_sims = len(targets)

    # grab n and num_snps from preprocessed dir
    if args.training_mean_sd is None:
        args.n, args.num_snps, meanSig, sdSig = np.load(
            args.out + "/preprocess_params.npy"
        )
    else:
        args.n, args.num_snps, meanSid, sdSig = np.load(args.training_mean_sd)
    args.n, args.num_snps = int(args.n), int(args.num_snps)

    # save training params:
    if args.pairs is None:
        args.pairs = int((args.n * (args.n - 1)) / 2)
    if args.pairs_encode is None:
        args.pairs_encode = int(args.pairs)
    np.save(
        args.out + "/Train/training_params_" + str(args.seed),
        [args.seed,
         args.max_epochs,
         args.validation_split,
         args.learning_rate,
         args.pairs,
         args.pairs_encode]
    )

    # split into val,train sets
    sim_ids = np.arange(0, total_sims)
    train, val = train_test_split(sim_ids, test_size=args.validation_split)
    if (
        len(val) * args.num_samples % args.batch_size != 0
        or len(train) * args.num_samples % args.batch_size != 0
    ):
        print(
            "\n\ntrain and val sets each need to be divisible by batch_size; \
            otherwise some batches will have missing data\n\n"
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
        shuffle_datasets=True,
        genos=genos,
        locs=locs,
        empirical_locs=None,
    )
    load_dl_modules()
    training_generator = DataGenerator(partition["train"], **params)
    validation_generator = DataGenerator(partition["validation"], **params)

    # train
    model, checkpointer, earlystop, reducelr = load_network()
    print("training!")
    model.fit(
        x=training_generator,
        epochs=args.max_epochs,
        shuffle=False,  # (redundant with shuffling inside the generator)
        verbose=args.keras_verbose,
        validation_data=validation_generator,
        callbacks=[checkpointer, earlystop, reducelr],
    )  # multi-thread via tf.config.threading.set_intra_op_parallelism_threads

    return


def predict():
    # grab mean and sd from training distribution
    if args.training_mean_sd is None:
        args.n, args.num_snps, meanSig, sdSig = np.load(
            args.out + "/preprocess_params.npy"
        )
    else:
        args.n, args.num_snps, meanSid, sdSig = np.load(args.training_mean_sd)
    args.n, args.num_snps = int(args.n), int(args.num_snps)

    # grab num pairs from saved training params
    params = np.load(args.out + "/Train/training_params_"
                     + str(args.seed) + ".npy")
    args.pairs, args.pairs_encode = int(params[4]), int(params[5])

    # load inputs
    targets, genos, locs = dict_from_preprocessed(args.out + "/Test/")
    total_sims = len(targets)

    # organize "partition" to hand to data generator
    partition = {}
    if args.num_pred is None:
        args.num_pred = int(total_sims)
    simids = np.random.choice(np.arange(total_sims),
                              args.num_pred,
                              replace=False)

    # get generator ready
    params = make_generator_params_dict(
        targets=targets,
        trees=None,
        shuffle_datasets=False,
        genos=genos,
        locs=locs,
        empirical_locs=None,
    )

    # predict
    print("predicting")
    outfile = args.out + "/Test/predictions_" + str(args.seed) + ".txt"
    if os.path.isfile(outfile):
        print("pred output exists; overwriting...")
        os.remove(outfile)
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()
    for b in range(
        int(np.ceil(args.num_pred / args.batch_size))
    ):  # loop to alleviate memory
        simids_batch = simids[b * args.batch_size:(b + 1) * args.batch_size]
        partition["prediction"] = np.array(simids_batch)
        generator = DataGenerator(partition["prediction"], **params)
        predictions = model.predict(generator)

        # unpack predictions
        with open(
            args.out + "/Test/predictions_" + str(args.seed) + ".txt", "a"
        ) as out_f:
            for i in range(len(predictions)):
                trueval = np.load(targets[simids_batch[i]])
                trueval = (trueval * sdSig) + meanSig
                trueval = np.exp(trueval)
                prediction = predictions[i][0]
                prediction = (prediction * sdSig) + meanSig
                prediction = np.exp(prediction)
                outline = "\t".join(map(str, [trueval, prediction]))
                print(outline, file=out_f)

    return


def empirical():
    # grab mean and sd from training distribution
    if args.training_mean_sd is None:
        args.n, args.num_snps, meanSig, sdSig = np.load(
            args.out + "/preprocess_params.npy"
        )
    else:
        args.n, args.num_snps, meanSid, sdSig = np.load(args.training_mean_sd)
    args.n, args.num_snps = int(args.n), int(args.num_snps)

    # grab num pairs from saved training params
    params = np.load(args.out + "/Train/training_params_"
                     + str(args.seed) + ".npy")
    args.pairs, args.pairs_encode = int(params[4]), int(params[5])

    # project locs
    locs = read_locs(args.empirical + ".locs")
    locs = np.array(locs)
    if len(locs) != args.n:
        print("length of locs file doesn't match n")
        exit()
    locs = project_locs(locs)

    # rescale locs
    locs = np.array(locs)
    minx = min(locs[:, 0])
    maxx = max(locs[:, 0])
    miny = min(locs[:, 1])
    maxy = max(locs[:, 1])
    x_range = maxx - minx
    y_range = maxy - miny
    locs[:, 0] = (locs[:, 0] - minx) / x_range  # rescale to (0,1)
    locs[:, 1] = (locs[:, 1] - miny) / y_range
    if x_range > y_range:  # these four lines for preserving aspect ratio
        locs[:, 1] *= y_range / x_range
    elif x_range < y_range:
        locs[:, 0] *= x_range / y_range

    # load model
    load_dl_modules()
    model, checkpointer, earlystop, reducelr = load_network()

    # convert vcf to geno matrix
    for i in range(args.num_reps):
        test_genos = vcf2genos(
            args.empirical + ".vcf", args.n, args.num_snps, args.phase
        )
        # (doesn't work if sample locations are repeated)
        # ibd(test_genos, locs, args.phase, args.num_snps)
        test_genos = np.reshape(
            test_genos, (1, test_genos.shape[0], test_genos.shape[1])
        )
        test_locs = np.reshape(locs, (1, locs.shape[1], locs.shape[0]))
        prediction = model.predict([test_genos, test_locs])

        # unpack_prediction
        with open(args.out + "/empirical_" + str(args.seed) + ".txt", "a") \
             as out_f:
            prediction = prediction[0][0]
            prediction = (prediction * sdSig) + meanSig
            prediction = np.exp(prediction)
            prediction = np.round(prediction, 10)
            print(args.empirical, "rep" + str(i), prediction, file=out_f)

    return


def plot_history():
    loss, val_loss = [], [
        np.nan
    ]  # loss starts at iteration 0; val_loss starts at end of first epoch
    with open(args.plot_history) as infile:
        for line in infile:
            if "val_loss:" in line:
                endofline = line.strip().split(" loss:")[-1]
                loss.append(float(endofline.split()[0]))
                val_loss.append(float(endofline.split()[3]))
    loss.append(np.nan)  # make columns even-length
    epochs = np.arange(len(loss))
    fig = plt.figure(figsize=(4, 1.5), dpi=200)
    plt.rcParams.update({"font.size": 7})
    ax1 = fig.add_axes([0, 0, 0.4, 1])
    ax1.plot(epochs, val_loss, color="blue", lw=0.5, label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.plot(epochs, loss, color="red", lw=0.5, label="loss")
    ax1.legend()
    fig.savefig(args.plot_history + "_plot.pdf", bbox_inches="tight")


def run():

    # main #
    np.random.seed(args.seed)

    # pre-process
    if args.preprocess is True:
        print("starting pre-processing pipeline")
        preprocess()

    # train
    if args.train is True:
        print("starting training pipeline")
        train()

    # plot training history
    if args.plot_history:
        plot_history()

    # predict
    if args.predict is True:
        print("starting prediction pipeline")
        if args.empirical is None:
            print("predicting on simulated data")
            predict()
        else:
            print("predicting on empirical data")
            empirical()
