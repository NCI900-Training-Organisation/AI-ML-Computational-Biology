import pyBigWig
import numpy as np
import re


def seq_to_hot(seq):
    seq = re.sub('B', 'N', seq)
    seq = re.sub('[D-F]', 'N', seq)
    seq = re.sub('[H-S]', 'N', seq)
    seq = re.sub('[U-Z]', 'N', seq)
    seq = seq.replace('a', 'A')
    seq = seq.replace('c', 'C')
    seq = seq.replace('g', 'G')
    seq = seq.replace('t', 'T')
    seq = seq.replace('n', 'N')
    Aseq = seq
    Aseq = Aseq.replace('A', '1')
    Aseq = Aseq.replace('C', '0')
    Aseq = Aseq.replace('G', '0')
    Aseq = Aseq.replace('T', '0')
    Aseq = Aseq.replace('N', '0')
    Aseq = np.asarray(list(Aseq), dtype='float32')
    Cseq = seq
    Cseq = Cseq.replace('A', '0')
    Cseq = Cseq.replace('C', '1')
    Cseq = Cseq.replace('G', '0')
    Cseq = Cseq.replace('T', '0')
    Cseq = Cseq.replace('N', '0')
    Cseq = np.asarray(list(Cseq), dtype='float32')
    Gseq = seq
    Gseq = Gseq.replace('A', '0')
    Gseq = Gseq.replace('C', '0')
    Gseq = Gseq.replace('G', '1')
    Gseq = Gseq.replace('T', '0')
    Gseq = Gseq.replace('N', '0')
    Gseq = np.asarray(list(Gseq), dtype='float32')
    Tseq = seq
    Tseq = Tseq.replace('A', '0')
    Tseq = Tseq.replace('C', '0')
    Tseq = Tseq.replace('G', '0')
    Tseq = Tseq.replace('T', '1')
    Tseq = Tseq.replace('N', '0')
    Tseq = np.asarray(list(Tseq), dtype='float32')
    hot = np.vstack((Aseq, Cseq, Gseq, Tseq))
    return hot


def random_shuffle(chrom_set, chr_len):
    tmp = []
    for the_chr in chrom_set:
        tmp.append(chr_len[the_chr])
    freq = np.rint(np.array(tmp) / sum(tmp) * 1000).astype('int')
    index_set = np.array([])

    for i in np.arange(len(chrom_set)):
        index_set = np.hstack((index_set, np.array([chrom_set[i]] * freq[i])))

    np.random.shuffle(index_set)

    return index_set


def generate_data_batch(if_train=True, if_test=False, the_tf=None, num_sample=10000, random_seed=30):

    path1 = '/g/data/ik06/stark/NCI_Leopard/data/dna_bigwig/'  # dna
    path2 = '/g/data/ik06/stark/NCI_Leopard/data/dnase_bigwig/'  # dnase
    path3 = '/g/data/ik06/stark/NCI_Leopard/data/chipseq_gem_bigwig/'  # label

    # open bigwig
    list_dna = ['A', 'C', 'G', 'T']
    dict_dna = {}
    for the_id in list_dna:
        dict_dna[the_id] = pyBigWig.open(path1 + the_id + '.bigwig')

    num_bp = np.array(
        [249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747,
         135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520,
         48129895, 51304566, 155270560])
    chr_all = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
               'chr13','chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

    chr_len = {}
    for i in np.arange(len(chr_all)):
        chr_len[chr_all[i]] = num_bp[i]

    size = 2 ** 11 * 5  # 10240
    num_channel = 5

    num_sample = int(num_sample)
    batch_size = 100

    chr_train_all = ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13',
                     'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr22', 'chrX']
    chr_test = ['chr1', 'chr21']
    ratio = 0.8
    np.random.seed(random_seed)
    np.random.shuffle(chr_train_all)
    tmp = int(len(chr_train_all) * ratio)
    chr_set1 = chr_train_all[:tmp]
    chr_set2 = chr_train_all[tmp:]
    print(chr_set1)
    print(chr_set2)
    print(chr_test)

    index_set1 = random_shuffle(chr_set1, chr_len)
    index_set2 = random_shuffle(chr_set2, chr_len)
    index_set3 = random_shuffle(chr_test, chr_len)

    i = 0
    genome_seq_batch = []
    label_batch = []

    for counter in range(num_sample):
        if if_test:
            if i == len(index_set3):
                i = 0
                np.random.shuffle(index_set3)
            the_chr = index_set3[i]
            i += 1
        elif if_train:
            if i == len(index_set1):
                i = 0
                np.random.shuffle(index_set1)
            the_chr = index_set1[i]
            i += 1
        else:
            if i == len(index_set2):
                i = 0
                np.random.shuffle(index_set2)
            the_chr = index_set2[i]
            i += 1



        feature_bw = pyBigWig.open(path2 + 'K562.bigwig')

        start = int(np.random.randint(0, chr_len[the_chr] - size, 1))
        end = start + size
        # print(the_chr, " START: ", start, " END: ", end)

        label_bw_list = []

        label_bw = pyBigWig.open(path3 + 'CTCF_K562.bigwig')
        label_bw_list.append(np.array(label_bw.values(the_chr, start, end)).T)
        label_bw.close()

        genome_seq = np.zeros((num_channel, size))
        num = 0
        for k in np.arange(len(list_dna)):
            the_id = list_dna[k]
            genome_seq[num, :] = dict_dna[the_id].values(the_chr, start, end)
            num += 1
        genome_seq[num, :] = np.nan_to_num(np.array(feature_bw.values(the_chr, start, end)), nan=0.0)

        genome_seq_batch.append(genome_seq.T)
        label = np.stack(label_bw_list, axis=-1)
        label_batch.append(label)

    return genome_seq_batch, label_batch

