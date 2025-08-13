# checking over the command line params

import os


def check_params(args):
    # avoid overwriting saved weights or other output files
    if args.train is True:
        if os.path.exists(
            args.out + "/Train/disperseNN2_" + str(args.seed) + ".weights.h5"
        ) and args.force is False:
            print("saved model with specified output name already \
                   exists. To force overwrite, use --force.")
            exit()
    if args.predict is True and args.empirical is None:
        if os.path.exists(args.out
                          + "/Test/predictions_"
                          + str(args.seed)
                          + ".txt") and args.force is False:
            print(
                "saved predictions with specified output name already exists. \
                To force overwrite, use --force."
            )
            exit()

    # other checks
    if (
        args.train is False
        and args.predict is False
        and args.preprocess is False
        and args.plot_history is False
        and args.empirical is None
    ):
        print(
            "either --help, --train, --predict, --preprocess,\
            --empirical,  or --plot_history"
        )
        exit()
    if args.train is True or args.predict is True or args.preprocess is True:
        if args.out is None:
            print("specify output directory --out")
            exit()
    if args.preprocess is True:
        if args.num_snps is None:
            print("specify num snps via --num_snps")
            exit()
        if args.n is None:
            print("specify sample size via --n")
            exit()
    if args.predict is True and args.empirical is None:
        if args.num_pred is not None:
            if args.num_pred % args.batch_size != 0:
                print(
                    "\n\npred sets each need to be divisible by batch_size; \
                    otherwise some batches will have missing data\n\n"
                )
                exit()
    if args.edge_width != "0" and args.empirical is not None:
        print(
            "can't specify edge width and empirical locations; at least not \
            currently"
        )
        exit()
