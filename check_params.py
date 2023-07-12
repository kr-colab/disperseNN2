# checking over the command line params
import os


def check_params(args):

    # avoid overwriting saved weights or other output files
    if args.train == True:
        if os.path.exists(args.out + "/Train/disperseNN2_" + str(args.seed) + "_model.hdf5"):
            print("saved model with specified output name already exists (i.e. --out)")
            exit()
        # if os.path.exists(f"{args.out}_training_mean_sd.npy"):
        #     print("saved mean and sd with specified output name already exists (i.e. --out)")
        #     exit()
    # if args.predict == True:
    #     if os.path.exists(f"{args.out}_predictions.txt"):
    #         print("saved predictions with specified output name already exists (i.e. --out)")
    #         exit()

    # arguments for training
    if args.train == True:
        if args.num_snps == None:
            print("specify num snps via --num_snps")
            exit()
        if args.n == None:
            print("specify sample size via --n")
            exit()

    # # arguments for prediction
    # if args.predict == True:
    #     if args.training_params == None:
    #         print("specify params file via --training_params")
    #         exit()

    # arguments related to preprocessing
    if args.preprocess == True:
        if args.num_reps > 1:
            print("can't bootstrap on preprocessed data, only tree sequences or VCF")
            exit()

    # check some other param combinations
    if args.train == False and args.predict == False and args.preprocess == False and args.plot_history == False:
        print("either --train or --predict or --preprocess")
        exit()
    if args.predict == True and args.empirical == None:
        if args.num_pred != None:
            if args.num_pred % args.batch_size != 0:
                print(
                    "\n\npred sets each need to be divisible by batch_size; otherwise some batches will have missing data\n\n"
                )
                exit()
    if args.predict == True and args.empirical == None:
        if args.n == None:
            print("missing sample size, via --n")
            exit()
    if args.edge_width != "0" and args.empirical != None:
        print("can't specify edge width and empirical locations; at least not currently")
        exit()
