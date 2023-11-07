import numpy as np
from Bio import SeqIO
from utils import debug_print

def read_genome(df, path="GRCh37_latest_genomic.fna"):
    tm = SeqIO.parse(path, "fasta")
    debug_print(["dna sequence imported from", path])

def generate_all_seqs(file_path="./data/seqs.txt", seq_len=23):
    """
    From the human genome, finds all possible sgRNA sequences ending with a NGG PAM.
    """

    genome_sequences = {}
    for record in SeqIO.parse("GRCh37_latest_genomic.fna", "fasta"):
        genome_sequences[record.id.split(".")[0]] = record.seq
        
    seq_file = open(file_path, "w")

    for i in range(1, 2): # Set to be only 1 so my computer doesnt explode
        try:
            if i < 10:
                chromosome_id = f"NC_00000{i}"
            else:
                chromosome_id = f"NC_0000{i}"
            genome_sequence = genome_sequences.get(chromosome_id).lower()
            
            for j in range(0, len(genome_sequence) - seq_len):
                seq = genome_sequence[j:j + seq_len]
                if seq[-2:] == "gg":
                    seq_file.write(f"{seq}\n")
        except:
            print(f"Could not find chromosome {chromosome_id} in genome.")

def ohe_bases(bases_lists):
    debug_print(["one-hot encoding bases"])
    ohe = np.zeros((len(bases_lists), len(bases_lists[0]), 4))
    for i, bases_list in enumerate(bases_lists):
        for j, base in enumerate(bases_list):
            if j >= len(bases_lists[0]):
                continue
            if base == "a":
                ohe[i, j, 0] = 1
            if base == "g":
                ohe[i, j, 1] = 1
            if base == "c":
                ohe[i, j, 2] = 1
            if base == "t":
                ohe[i, j, 3] = 1
    return ohe

def ohe_epi(epi_lists):
    debug_print(["one-hot encoding epigenetic features"])
    ohe = np.zeros((len(epi_lists), len(epi_lists[0]), 4))
    return ohe

def noise_data(x, noise_level=0.1):
    """
    Add noise to the input data.
    """
    noise = np.random.normal(loc=0.0, scale=noise_level, size=x.shape)
    x_noisy = x + noise
    x_noisy = np.clip(x_noisy, 0., 1.)
    return x_noisy


def get_ae_data(file_path="./data/seqs.txt", validation_split = 0.2):
    """
    Load the data for the autoencoder.
    """

    # Read sequences
    seqs = []
    with open(file_path, "r") as f:
        for line in f:
            seqs.append(line.strip())

    # For testing purpose, only use first 10000 sequences
    seqs = seqs[:10000]

    # Convert to numpy array
    seqs = np.array(seqs)
    seqs = seqs.reshape(-1, 1)

    # Split into bases
    seqs = np.char.array(seqs)
    seqs = np.array([list(seq[0]) for seq in seqs])

    # OHE
    seqs = ohe_bases(seqs)

    # Append epi data
    # TODO: Implement epi data
    epi = ohe_epi(seqs)
    x = np.concatenate((seqs, epi), axis=2)
    
    # Add noise   
    y = noise_data(x)

    # Split into train and test sets
    split = int(len(x) * validation_split)
    x_train = x[:-split]
    y_train = y[:-split]
    x_test = x[-split:]
    y_test = y[-split:]

    return x_train, y_train, x_test, y_test

def get_on_target_data(file_path="./data/on_target.txt", validation_split = 0.2):
    # TODO: Use real data

    # For now, just make up data of right shape
    x = np.random.rand(10000, 23, 8)
    y = np.random.rand(10000, 1)
    y = np.round(y)

    # Split into train and test sets
    split = int(len(x) * validation_split)
    x_train = x[:-split]
    y_train = y[:-split]
    x_test = x[-split:]
    y_test = y[-split:]
    
    return x_train, y_train, x_test, y_test

def get_on_target_reg_data(file_path="./data/on_target_reg.txt", validation_split = 0.2):
    # TODO: Use real data

    # For now, just make up data of right shape
    x = np.random.rand(10000, 23, 8)
    y = np.random.rand(10000, 1)

    # Split into train and test sets
    split = int(len(x) * validation_split)
    x_train = x[:-split]
    y_train = y[:-split]
    x_test = x[-split:]
    y_test = y[-split:]
    
    return x_train, y_train, x_test, y_test

def get_off_target_data(file_path="./data/off_target.txt", validation_split = 0.2):
    # TODO: Use real data

    # For now, just make up data of right shape
    # Unlike off target, input is a tuple of on target and off target sequences
    target_seqs = np.random.rand(10000, 23, 8)
    off_target_seqs = np.random.rand(10000, 23, 8)
    y = np.random.rand(10000, 1)
    y = np.round(y)

    # Split into train and test sets
    split = int(len(target_seqs) * (1 - validation_split))
    x_train = [target_seqs[:split], off_target_seqs[:split]]
    y_train = y[:split]
    x_test = [target_seqs[split:], off_target_seqs[split:]]
    y_test = y[split:]

    return x_train, y_train, x_test, y_test


def get_off_target_reg_data(file_path="./data/off_target_reg.txt", validation_split = 0.2):
    # TODO: Use real data

    # For now, just make up data of right shape
    # Unlike off target, input is a tuple of on target and off target sequences
    target_seqs = np.random.rand(10000, 23, 8)
    off_target_seqs = np.random.rand(10000, 23, 8)
    y = np.random.rand(10000, 1)

    # Split into train and test sets
    split = int(len(target_seqs) * (1 - validation_split))
    x_train = [target_seqs[:split], off_target_seqs[:split]]
    y_train = y[:split]
    x_test = [target_seqs[split:], off_target_seqs[split:]]
    y_test = y[split:]

    return x_train, y_train, x_test, y_test
