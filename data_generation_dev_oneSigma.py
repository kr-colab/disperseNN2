# data generator code for training CNN

import sys
import numpy as np
import tensorflow as tf
import msprime
import tskit
import warnings
from attrs import define,field
from read_input import *
import itertools

#import psutil # for printing memory usage for bug                                                                                                                            
#import os # for print memory usage
import gc # garbage collect

@define
class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    list_IDs: list
    targets: dict 
    trees: dict
    num_snps: int
    min_n: int
    max_n: int
    batch_size: int
    mu: float
    threads: int
    shuffle: bool
    rho: float
    baseseed: int
    recapitate: bool
    mutate: bool
    crop: float
    sampling_width: float
    edge_width: dict
    phase: int
    polarize: int
    sample_widths: dict
    genos: dict
    locs: dict
    preprocessed: bool
    num_reps: int
    combination_size: int
    grid_coarseness: int
    segment: bool
    sample_grid: int


    def __attrs_post_init__(self):
        "Initialize a few things"
        self.on_epoch_end()
        np.random.seed(self.baseseed)
        warnings.simplefilter("ignore", msprime.TimeUnitsMismatchWarning) # (recapitate step)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
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
                loc[0] > (i*bin_size)
                and loc[0] < ((i+1)*bin_size)
                and loc[1] > (j*bin_size)
                and loc[1] < ((j+1)*bin_size)
            ):
                output_ind = ind
                break
        if output_ind == None: # if no individuals in the current square, then choose a random ind.
            output_ind = np.random.choice(sampled_inds, 1, replace=False)                          

        return output_ind


    def unpolarize(self, snp, n):
        "Change 0,1 encoding to major/minor allele. Also filter no-biallelic"
        alleles = {}                                                                          
        for i in range(n * 2):  
            a = snp[i]                                                               
            if a not in alleles:                                                              
                alleles[a] = 0                                                                
            alleles[a] += 1                                                                   
        if len(alleles) == 2:                                                                 
            new_genotypes = []                                                                
            major, minor = list(set(alleles))  # set() gives random order                     
            if alleles[major] < alleles[minor]:                                               
                major, minor = minor, major                                                   
            for i in range(n * 2):  # go back through and convert genotypes                   
                a = snp[i]                                                           
                if a == major:                                                                
                    new_genotype = 0                                                          
                elif a == minor:                                                              
                    new_genotype = 1                                                          
                new_genotypes.append(new_genotype)
        else:
            new_genotypes = False
            
        return new_genotypes

    
    def sample_ts(self, filepath, seed):
        "The meat: load in and fully process a tree sequence"

        # read input                        
        ts = tskit.load(filepath)
        np.random.seed(seed)

        # grab map width and sigma from provenance
        #W = parse_provenance(ts, 'W')
        W = 50 # ***
        if self.edge_width == 'sigma':
            edge_width = parse_provenance(ts, 'sigma')
        else:
            edge_width = float(self.edge_width)

        # recapitate
        alive_inds = []
        for i in ts.individuals():
            alive_inds.append(i.id)
        if self.recapitate == "True":
            Ne = len(alive_inds)
            if ts.num_populations > 1:
                ts = ts.simplify() # gets rid of weird, extraneous populations
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
        if self.sampling_width != None:
            sample_width = (float(self.sampling_width) * W) - (edge_width * 2)
        else:
            sample_width = np.random.uniform(
                0, W - (edge_width * 2)
            ) 
        n = np.random.randint(
            self.min_n, self.max_n + 1
        )  # (excludes the max of the range)
        sampled_inds = self.cropper(ts, W, sample_width, edge_width, alive_inds)
        failsafe = 0
        while (
            len(sampled_inds) < n
        ):  # keep looping until you get a map with enough samples
            if self.sampling_width != None:
                sample_width = (float(self.sampling_width) * W) - (edge_width * 2)
            else:
                sample_width = np.random.uniform(0, W - (edge_width * 2))
            n = np.random.randint(
                self.min_n, self.max_n + 1
            )  # (excludes the max of the range)
            failsafe += 1
            if failsafe > 100:
                print("\tnot enough samples, killed while-loop after 100 loops")
                sys.stdout.flush()
                exit()
            sampled_inds = self.cropper(ts, W, sample_width, edge_width, alive_inds)

        # sample individuals
        if self.sample_grid == None:
            keep_indivs = np.random.choice(sampled_inds, n, replace=False)
        else:
            if n < self.sample_grid**2:
                print("your sample grid is too fine, not enough samples to fill it")
                exit()
            keep_indivs = []
            for r in range(int(np.ceil(self.max_n / self.sample_grid**2))): # sampling from each square multiple times until >= n samples
                for i in range(self.sample_grid):
                    for j in range(self.sample_grid):
                        new_guy = self.sample_ind(ts,sampled_inds,W,i,j)                    
                        keep_indivs.append(new_guy)
                        sampled_inds.remove(new_guy) # to avoid sampling same guy twice
            keep_indivs = np.random.choice(keep_indivs, n, replace=False) # taking n from the >=n list            
        keep_nodes = []
        for i in keep_indivs:
            ind = ts.individual(i)
            keep_nodes.extend(ind.nodes)

        # simplify
        ts = ts.simplify(keep_nodes)

        # mutate
        if self.num_reps == 1:
            total_snps = self.num_snps
        else:
            total_snps = self.num_snps * 10 # arbitrary size of SNP table for bootstraps
        if self.mutate == "True":
            mu = float(self.mu)
            ts = msprime.sim_mutations(
                ts,
                rate=mu,
                random_seed=seed,
                model=msprime.SLiMMutationModel(type=0),
                keep=True,
            )
            counter = 0
            while ts.num_sites < (total_snps * 2): # extra SNPs because a few are likely  non-biallelic
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

        # find width of sampling area and save dists
        if self.combination_size == 2:
            locs = np.array(locs)
            sampling_width = 0
#            dists = []
            for i in range(0,n-1):
                for j in range(i+1,n):
                    d = ( (locs[i,0]-locs[j,0])**2 + (locs[i,1]-locs[j,1])**2 )**(1/2)
#                    dists.append(d)
                    if d > sampling_width:
                        sampling_width = float(d)

        # elif self.combination_size == 3: # this doesn't have the desired affect on pairs, haven't thought about bigger than trios
        #     combinations = list(itertools.combinations(range(self.max_n), self.combination_size))
        #     num_pairs = int( self.combination_size * (self.combination_size-1) / 2)
        #     locs = np.array(locs)
        #     sampling_width = 0
        #     dists = []
        #     for comb in combinations:
        #         current_dists = []
        #         for pair in list(itertools.combinations(range(num_pairs), 2)):
        #             d = ( (locs[comb[pair[0]],0]-locs[comb[pair[1]],0])**2 + (locs[comb[pair[0]],1]-locs[comb[pair[1]],1])**2 )**(1/2)  
        #             current_dists.append(d)
        #             if d > sampling_width:
        #                 sampling_width = float(d)          
        #         dists.append(current_dists)
        # dists = np.array(dists)

        # # rescale locs
        # locs0 = np.array(locs)
        # minx = min(locs0[:, 0])
        # maxx = max(locs0[:, 0])
        # miny = min(locs0[:, 1])
        # maxy = max(locs0[:, 1])
        # x_range = maxx - minx
        # y_range = maxy - miny
        # locs0[:, 0] = (locs0[:, 0] - minx) / x_range  # rescale to (0,1)      
        # locs0[:, 1] = (locs0[:, 1] - miny) / y_range
        # if   x_range > y_range: # these four lines for preserving aspect ratio
        #     locs0[:, 1] *= y_range / x_range
        # elif x_range < y_range:
        #     locs0[:, 0] *= x_range / y_range
        # #locs = locs0.T 

        # # grab pairwise locations                                                                                                                                          
        # pairwise_locs = []
        # for i in range(0,n-1):
        #     for j in range(i+1,n):
        #         pairwise_locs.append( np.concatenate( (locs[i,:],locs[j,:]) ))

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
                new_genotypes = self.unpolarize(geno_mat0[shuffled_indices[s]], n)
                if new_genotypes != False: # if bi-allelic, add in the snp
                    geno_mat1.append(new_genotypes)            
                    snp_index_map[shuffled_indices[s]] = int(snp_counter)
                    snp_counter += 1
            while snp_counter < total_snps: # likely need to replace a few non-biallelic sites
                s += 1
                new_genotypes = self.unpolarize(geno_mat0[shuffled_indices[s]], n)
                if new_genotypes != False:
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
            geno_mat1 = np.zeros((total_snps, n))
            for ind in range(n):
                geno_mat1[:, ind] += geno_mat0[:, ind * 2]
                geno_mat1[:, ind] += geno_mat0[:, ind * 2 + 1]
            geno_mat0 = np.array(geno_mat1) # (change variable name)

        # sample SNPs
        mask = [True] * self.num_snps + [False] * (total_snps - self.num_snps)
        np.random.shuffle(mask)
        geno_mat1 = geno_mat0[mask, :]
        geno_mat2 = np.zeros((self.num_snps, self.max_n * self.phase)) # pad
        geno_mat2[:, 0 : n * self.phase] = geno_mat1
        
        # # garbage collect, print memory
        # process = psutil.Process(os.getpid())
        # with open("Boxes75/job201_gc.txt", "a") as f:
        #     f.write(str(process) + "," + filepath + "\n")
        #     f.write("before" + "\n")
        #     f.write(str(process.memory_info().rss / 1024 ** 2) + "\n")
        #     f.write(str(process.memory_info().vms / 1024 ** 2) + "\n")

        #     gc.collect()

        #     f.write("after" + "\n")
        #     f.write(str(process.memory_info().rss / 1024 ** 2) + "\n")
        #     f.write(str(process.memory_info().vms / 1024 ** 2) + "\n")
            
        del ts
        del geno_mat0
        del geno_mat1
        del mask
        gc.collect()
        
        return geno_mat2, locs.T # not re-scaled... because I'm not giving it the sampling area currently



    # def preprocessed_sample_ts(self, geno_path): 
    #     "Seperate function for loading in pre-processed data"
        
    #     # read input
    #     geno_mat = np.load(geno_path)
    #     loc_mat = np.load(_path)
    #     target_mat = np.load(geno_path)

    #     return geno_mat

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"        
        X1 = np.empty((self.batch_size, self.num_snps, self.max_n))  # genos
        X2 = np.empty((self.batch_size, 2, self.max_n))  # locs                                 
        if self.segment == False:
            y = np.empty((self.batch_size, ), dtype=float)  # targets      
        else:
            y_reg = np.empty((self.batch_size, 500, 500), dtype=float)
            y_class = np.empty((self.batch_size, 500, 500, 4), dtype=float)
            targets_reg = self.targets[0] # * this is what I did before and it worked... seem weird that it's outside the loop. *  t
            targets_class = self.targets[1]
        
        if self.preprocessed == False:
            for i, ID in enumerate(list_IDs_temp):
                if self.segment == False:
                    y[i] = self.targets[ID]
                else:
                    y_reg[i] = targets_reg[ID]
                    y_class[i] = targets_class[ID]
                out = self.sample_ts(self.trees[ID], np.random.randint(1e9,size=1))
                X1[i, :] = out[0]
                X2[i, :] = out[1]
                X = [X1, X2]
        else:
            for i, ID in enumerate(list_IDs_temp):
                if self.segment == False:
                    y[i] = np.load(self.targets[ID])
                else:
                    y_both = np.load(self.targets[ID])
                    y_reg[i] = y_both[:,:,0]
                    y_class[i] = y_both[:,:,1:5]
                X1[i, :] = np.load(self.genos[ID])
                X2[i, :] = np.load(self.locs[ID])
                X = [X1, X2]

        if self.segment == True:
            y =[y_reg, y_class]

        return (X, y)