# data generator code for training disperseNN2

import sys
import numpy as np
import tensorflow as tf
import msprime
import tskit
import warnings
from attrs import define
from disperseNN2.read_input import parse_provenance
import gc


@define
class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    list_IDs: list
    targets: dict
    trees: dict
    num_snps: int
    n: int
    batch_size: int
    mu: float
    shuffle_datasets: bool
    shuffle_individuals: bool
    rho: float
    baseseed: int
    recapitate: bool
    skip_mutate: bool
    edge_width: str
    phase: int
    polarize: int
    genos: dict
    locs: dict
    grid_coarseness: int
    sample_grid: int
    empirical_locs: list

    def __attrs_post_init__(self):
        "Initialize a few things"
        self.on_epoch_end()
        np.random.seed(self.baseseed)
        warnings.simplefilter(
            "ignore", msprime.TimeUnitsMismatchWarning
        )  # (recapitate step)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:
                               (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle_datasets is True:
            np.random.shuffle(self.indexes)

    def cropper(self, ts, W, sample_width, edge_width, alive_inds):
        "Cropping the map, returning individuals inside sampling window"
        cropped = []
        left_edge = np.random.uniform(
            low=edge_width, high=W - edge_width - sample_width
        )
        right_edge = left_edge + sample_width
        bottom_edge = np.random.uniform(
            low=edge_width, high=W - edge_width - sample_width
        )
        top_edge = bottom_edge + sample_width

        for i in alive_inds:
            ind = ts.individual(i)
            loc = ind.location[0:2]
            if (
                loc[0] > left_edge
                and loc[0] < right_edge
                and loc[1] > bottom_edge
                and loc[1] < top_edge
            ):
                cropped.append(i)

        return cropped

    def sample_ind(self, ts, sampled_inds, W, i, j):
        bin_size = W / self.sample_grid
        output_ind = None
        for ind in sampled_inds:
            indiv = ts.individual(ind)
            loc = indiv.location[0:2]
            if (
                loc[0] > (i * bin_size)
                and loc[0] < ((i + 1) * bin_size)
                and loc[1] > (j * bin_size)
                and loc[1] < ((j + 1) * bin_size)
            ):
                output_ind = ind
                break
        if (
            output_ind is None
        ):  # if no individuals in the current square, choose a random ind
            output_ind = np.random.choice(sampled_inds, 1, replace=False)

        return output_ind

    def unpolarize(self, snp):
        "Change 0,1 encoding to major/minor allele. Also filter no-biallelic"
        alleles = {}
        for i in range(self.n * 2):
            a = snp[i]
            if a not in alleles:
                alleles[a] = 0
            alleles[a] += 1
        if len(alleles) == 2:
            new_genotypes = []
            major, minor = list(set(alleles))  # set() gives random order
            if alleles[major] < alleles[minor]:
                major, minor = minor, major
            for i in range(self.n * 2):  # go back through, convert genotypes
                a = snp[i]
                if a == major:
                    new_genotype = 0
                elif a == minor:
                    new_genotype = 1
                new_genotypes.append(new_genotype)
        else:
            new_genotypes = False

        return new_genotypes

    def empirical_sample(self, ts, sampled_inds, n, N, W):
        locs = np.array(self.empirical_locs)
        np.random.shuffle(locs)
        indiv_dict = {}  # tracking which indivs have been picked up already
        for i in sampled_inds:
            indiv_dict[i] = 0
        keep_indivs = []
        for pt in range(n):  # for each sampling location
            dists = {}
            for i in indiv_dict:
                ind = ts.individual(i)
                loc = ind.location[0:2]
                d = ((loc[0] - locs[pt, 0]) ** 2
                     + (loc[1] - locs[pt, 1]) ** 2) ** (
                         1 / 2
                     )
                dists[d] = i  # see what I did there?
            nearest = dists[min(dists)]
            ind = ts.individual(nearest)
            loc = ind.location[0:2]
            keep_indivs.append(nearest)
            del indiv_dict[nearest]

        return keep_indivs

    def sample_ts(self, filepath, seed):
        "The meat: load in and fully process a tree sequence"

        # read input
        ts = tskit.load(filepath)
        np.random.seed(seed)

        # grab map width and sigma from provenance
        W = parse_provenance(ts, "W")
        if self.edge_width == "sigma":
            edge_width = parse_provenance(ts, "sigma")
        else:
            edge_width = float(self.edge_width)

        # recapitate
        alive_inds = []
        for i in ts.individuals():
            alive_inds.append(i.id)
        if self.recapitate == "True":
            Ne = len(alive_inds)
            if ts.num_populations > 1:
                ts = ts.simplify()  # gets rid of weird, extraneous populations
            demography = msprime.Demography.from_tree_sequence(ts)
            demography[0].initial_size = Ne
            ts = msprime.sim_ancestry(
                initial_state=ts,
                recombination_rate=self.rho,
                demography=demography,
                start_time=ts.metadata["SLiM"]["generation"],
                random_seed=seed,
            )

        # crop map
        sample_width = W - (edge_width * 2)
        sampled_inds = self.cropper(ts,
                                    W,
                                    sample_width,
                                    edge_width,
                                    alive_inds)
        if len(sampled_inds) < self.n:
            print("\tnot enough samples, killed while-loop after 100 loops",
                  flush=True)
            exit()

        # sample individuals
        if self.sample_grid is not None:
            if self.n < self.sample_grid**2:
                print("your sample grid is too fine, \
                not enough samples to fill it")
                exit()
            keep_indivs = []
            for r in range(
                int(np.ceil(self.n / self.sample_grid**2))
            ):  # sampling from each square multiple times until >= n samples
                for i in range(self.sample_grid):
                    for j in range(self.sample_grid):
                        new_guy = self.sample_ind(ts, sampled_inds, W, i, j)
                        keep_indivs.append(new_guy)
                        sampled_inds.remove(new_guy)  # avoid sampling same guy
            keep_indivs = np.random.choice(
                keep_indivs, self.n, replace=False
            )  # taking n from the >=n list
        elif self.empirical_locs is not None:
            keep_indivs = self.empirical_sample(
                ts, sampled_inds, self.n, len(sampled_inds), W
            )
        else:
            keep_indivs = np.random.choice(sampled_inds, self.n, replace=False)
        # (unindent)
        keep_nodes = []
        for i in keep_indivs:
            ind = ts.individual(i)
            keep_nodes.extend(ind.nodes)

        # simplify
        ts = ts.simplify(keep_nodes)

        # mutate
        total_snps = self.num_snps
        if self.skip_mutate is False:
            mu = float(self.mu)
            ts = msprime.sim_mutations(
                ts,
                rate=mu,
                random_seed=seed,
                model=msprime.SLiMMutationModel(type=0),
                keep=True,
            )
            counter = 0
            while ts.num_sites < (
                total_snps * 2
            ):  # extra SNPs because a few are likely  non-biallelic
                counter += 1
                mu *= 10
                ts = msprime.sim_mutations(
                    ts,
                    rate=mu,
                    random_seed=seed,
                    model=msprime.SLiMMutationModel(type=0),
                    keep=True,
                )
                if counter == 10:
                    print("\n\nsorry, Dude. Didn't generate enough snps. \n\n")
                    sys.stdout.flush()
                    exit()

        # grab spatial locations
        sample_dict = {}
        locs = []
        for samp in ts.samples():
            node = ts.node(samp)
            indID = node.individual
            if indID not in sample_dict:
                sample_dict[indID] = 0
                loc = ts.individual(indID).location[0:2]
                locs.append(loc)

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
        locs = locs.T

        # grab genos
        geno_mat0 = ts.genotype_matrix()

        # change 0,1 encoding to major/minor allele
        if self.polarize == 2:
            shuffled_indices = np.arange(ts.num_sites)
            np.random.shuffle(shuffled_indices)
            geno_mat1 = []
            snp_counter = 0
            snp_index_map = {}
            for s in range(total_snps):
                new_genotypes = self.unpolarize(geno_mat0[shuffled_indices[s]])
                if new_genotypes is not False:  # if bi-allelic, add in the snp
                    geno_mat1.append(new_genotypes)
                    snp_index_map[shuffled_indices[s]] = int(snp_counter)
                    snp_counter += 1
            while (
                snp_counter < total_snps
            ):  # likely need to replace a few non-biallelic sites
                s += 1
                new_genotypes = self.unpolarize(geno_mat0[shuffled_indices[s]])
                if new_genotypes is not False:
                    geno_mat1.append(new_genotypes)
                    snp_index_map[shuffled_indices[s]] = int(snp_counter)
                    snp_counter += 1
            geno_mat0 = []
            sorted_indices = list(snp_index_map)
            sorted_indices.sort()
            for snp in range(total_snps):
                geno_mat0.append(geno_mat1[snp_index_map[sorted_indices[snp]]])
            geno_mat0 = np.array(geno_mat0)

        # sample SNPs
        else:
            mask = [True] * total_snps + [False] * (ts.num_sites - total_snps)
            np.random.shuffle(mask)
            geno_mat0 = geno_mat0[mask, :]

        # collapse genotypes, change to minor allele dosage (e.g. 0,1,2)
        if self.phase == 1:
            geno_mat1 = np.zeros((total_snps, self.n))
            for ind in range(self.n):
                geno_mat1[:, ind] += geno_mat0[:, ind * 2]
                geno_mat1[:, ind] += geno_mat0[:, ind * 2 + 1]
            geno_mat0 = np.array(geno_mat1)  # (change variable name)

        # sample SNPs
        mask = [True] * self.num_snps + [False] * (total_snps - self.num_snps)
        np.random.shuffle(mask)
        geno_mat1 = geno_mat0[mask, :]
        geno_mat2 = np.zeros((self.num_snps, self.n * self.phase))  # pad
        geno_mat2[:, 0:self.n * self.phase] = geno_mat1

        # free memory
        del ts
        del geno_mat0
        del geno_mat1
        del mask
        gc.collect()

        return geno_mat2, locs

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"
        X1 = np.empty((self.batch_size, self.num_snps, self.n), dtype="int8")
        X2 = np.empty((self.batch_size, 2, self.n), dtype=float)
        y = np.empty((self.batch_size,), dtype=float)
        shuffled_indices = np.arange(self.n)
        if self.shuffle_individuals is True: # augment training set by shuffling individuals
            np.random.shuffle(shuffled_indices)
        for i, ID in enumerate(list_IDs_temp):
            # load target
            y[i] = np.load(self.targets[ID])

            # load and re-order genos
            genomat = np.load(self.genos[ID])
            if self.empirical_locs is None:
                genomat = genomat[
                    :, shuffled_indices
                ]
            X1[i, :] = genomat

            # load and re-order locs
            locs = np.load(self.locs[ID])
            if self.empirical_locs is None:
                locs = locs[
                    :, shuffled_indices
                ]
            X2[i, :] = locs

        # (unindent)
        X = [X1, X2]

        return (X, y)
