import sys
import numpy as np
import gzip
import random

# params
vcf_path = sys.argv[1]
sample_path = sys.argv[2]  # list of samples to keep
outname = sys.argv[3]  # with .vcf
min_read_depth = int(sys.argv[4])  # min read depth, DP field
min_sample_prop = float(sys.argv[5])  # min proportion of samples with data
seed = int(sys.argv[6])

# set seed
# np.random.seed(seed)
random.seed(seed)  # currently affects lines with MAF=0.5, 0s and 1s random

# read in samples
keepers = {}
with open(sample_path) as infile:
    for line in infile:
        keepers[line.strip().split(",")[0]] = 0
n = len(keepers)


# open vcfs
vcf = gzip.open(vcf_path, "r")
filtered_vcf = gzip.open(outname + ".gz", "w")


# other initializations
nucs = ["A", "T", "C", "G", "<NON_REF>"]  # (last one due to suflower)
lc = ["a", "t", "c", "g"]


# parse vcf
for line in vcf:
    line = line.decode("UTF-8")
    if line[0:2] == "##":
        filtered_vcf.write(line.encode("utf-8"))
    elif line[0] == "#":
        header = line.strip().split()
        keep_samples = []
        for i in range(len(header)):
            if header[i] in keepers:
                keep_samples.append(i)
        header = np.array(header)
        header = np.concatenate((header[0:9], header[keep_samples]))
        filtered_vcf.write(("\t".join(header) + "\n").encode("utf-8"))
    else:
        currentline = line.strip().split("\t")

        # filter: indels
        indel = False
        ref, alt = currentline[3:5]
        refs = ref.split(",")
        alts = alt.split(",")
        for r in refs:
            if r not in nucs and r != "":  # tskit outputs "" for ref
                try:
                    int(r)
                except:
                    indel = True
        for a in alts:
            if a not in nucs:
                try:
                    int(a)
                except:
                    indel = True

        # check genotype field identifiers (pretty wonky in some cases)
        identifiers = currentline[8].split(":")
        try:
            GT_index = identifiers.index("GT")
        except:
            GT_index = None
        try:
            DP_index = identifiers.index("DP")
        except:
            DP_index = None  # tskit doesn't have DP...
        if indel is False and GT_index is not None:
            out_genos = []  # this will be a line in the output vcf
            ac = {}  # allele dictionary
            missing_count = 0  # counting missing data
            for sample in keep_samples:
                # rarely in AG1000 we get truncted information
                if currentline[sample] == "./.":
                    currentline[sample] = "./.:0,0,0:0:0"
                # sunflower data has some crazy DP fields (beyond just ".")
                if DP_index is not None:
                    DP = currentline[sample].split(":")[DP_index]
                else:
                    DP = 999
                try:
                    int(DP)
                except:
                    currentline[sample] = "./.:0,0,0:0:0"
                    DP = 0

                # getting rid of phased bar ("|")
                all_geno_fields = currentline[sample].split(":")
                genotype = str(all_geno_fields[GT_index])
                if "|" in genotype:
                    genotype = genotype.replace("|", "/")
                    all_geno_fields[GT_index] = str(genotype)
                    currentline[sample] = ":".join(all_geno_fields)

                # require e.g. DP>=10 to genotype
                if int(DP) < min_read_depth or genotype == "./.":
                    currentline[sample] = "./.:0,0,0:0:0"
                    missing_count += 1

                # quick check if missing data
                alleles = genotype.split("/")
                if alleles == [".", "."]:
                    pass

                # checking for other potential garbage
                else:
                    for a in alleles:
                        try:
                            int(a)
                        except:
                            sys.stderr.write(
                                "messed up allele:"
                                + ",".join(currentline[0:2])
                                + currentline[sample]
                                + " \n"
                            )
                            print("messed up allele:",
                                  currentline[0:2],
                                  genotype)
                            exit()

                        # keeping track of alleles
                        if a not in ac:
                            ac[a] = 0
                        ac[a] += 1

                # append to growing line
                out_genos.append(currentline[sample])

            # filter non-biallelic
            different_genos = list(set(ac))
            if len(different_genos) != 2:
                pass

            # figure out major/minor alleles
            else:
                anc, der = list(set(ac))  # set() gives random order
                if ac[anc] < ac[der]:
                    anc, der = der, anc
                anc = str(anc)
                der = str(der)

                # convert genotypes accordingly
                # what do I want to do here.
                # I want to convert 0,1,2,3 alleles to 0,1, by major allele
                for s in range(n):
                    genotype = out_genos[s].split(":")[0]
                    split = genotype.split("/")
                    new_genotype = []
                    for al in split:
                        if al == anc:
                            new_genotype.append("0")
                        elif al == der:
                            new_genotype.append("1")
                        elif al == ".":
                            new_genotype.append(
                                "0"
                            )  # ***** ***** IMPUTATION ***** *****
                        else:
                            print("somethings broken")
                            exit()
                    out_genos[s] = "/".join(new_genotype)

                # filter sample representation
                missing_prop = float(missing_count) / float(n)
                represented = 1 - missing_prop
                if represented < min_sample_prop:
                    pass

                # output
                else:
                    outline = currentline[0:9] + out_genos
                    filtered_vcf.write(("\t".join(outline)
                                        + "\n").encode("utf-8"))


# close files
vcf.close()
filtered_vcf.close()
