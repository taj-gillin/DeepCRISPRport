import pandas as pd
import numpy as np
from Bio import SeqIO
from utils import *
import csv
from tqdm import tqdm

def read_genome(df, path="GRCh37_latest_genomic.fna"):
    tm = SeqIO.parse(path, "fasta")
    debug_print(["dna sequence imported from", path])

def generate_all_seqs(genome_file_path, output_file_path="./data/seqs_with_positions.csv", seq_len=23):
    """
    From the human genome, finds all possible sgRNA sequences ending with a NGG PAM,
    and stores them along with their start and end positions.
    """
    genome_sequences = {}
    for record in SeqIO.parse(genome_file_path, "fasta"):
        if record.id.split(".")[0][:2] == "NC":
            genome_sequences[record.id.split(".")[0]] = record.seq

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['sgRNA', 'chromosome', 'Start', 'End'])  # Columns for sgRNA sequence and its positions

        for chromosome_id, genome_sequence in genome_sequences.items():
            debug_print([f"Processing chromosome {chromosome_id}"])
            genome_sequence = genome_sequence.lower()

            for j in range(0, len(genome_sequence) - seq_len):
                if genome_sequence[j + seq_len - 2:j + seq_len] == "gg":
                    seq = genome_sequence[j:j + seq_len - 2]
                    start = j
                    end = j + seq_len
                    writer.writerow([seq, chromosome_id, start, end])


def noise_data(x, noise_level=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=x.shape)
    x_noisy = x + noise
    x_noisy = np.clip(x_noisy, 0., 1.)
    return x_noisy

def get_ae_data(file_path = "./data/seqs.csv", max_seqs = -1):
    debug_print(["Reading data from", file_path])
    df = pd.read_csv(file_path)

    # Initialize x as a list to hold sgRNA sequences with their epigenomic data
    x = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Stop if we have enough sequences
        if index >= max_seqs and max_seqs != -1:
            break

        # One-hot encode the sgRNA sequence
        try:
            one_hot_seq = get_seqcode(row['sgRNA'] + "gg")
        except:
            debug_print(["Error with sequence: ", row['sgRNA']])
            continue

        # Fetch the epigenomic data
        epi_data = get_epi_data(row['chromosome'], row['Start'], row['End'] - 1)

        # Concatenate sequence data with epigenomic data
        x.append(np.concatenate([one_hot_seq, epi_data], axis=-1))

    # Convert x to a numpy array
    x = np.array(x)
    x = x.reshape(-1, 23, 8)

    y = np.copy(x)
    x = noise_data(x)

    return x, y

def get_ontar_data(file_path, reg=False):
    # Read the dataset
    df = pd.read_csv(file_path)

    # Process x data
    combined_data = []

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        one_hot_seq = get_seqcode(row['sgRNA'])
        epi_data = get_epi_data(row['Chromosome'], row['Start'], row['End'])
        combined_seq = np.concatenate((one_hot_seq, epi_data), axis=2)
        combined_data.append(combined_seq)

    x = np.array(combined_data)        
    x = x.reshape(-1, 23, 8)

    # Process y data
    y = df['Normalized efficacy'].values.reshape(-1, 1)

    # If classification, round the efficacy values
    if not reg:
        y = np.round(y)

    return x, y

def get_offtar_data(on_target_file_path, off_target_file_path, reg=False):
    on_target_df = pd.read_csv(on_target_file_path)
    off_target_df = pd.read_csv(off_target_file_path)

    # Merge the off-target data with the corresponding on-target data
    merged_df = off_target_df.merge(on_target_df, left_on='Target ID', right_on='Target ID', suffixes=('_off', '_on'))

    combined_on_data = []
    combined_off_data = []

    for idx, row in merged_df.iterrows():
        # Process on-target sequence
        one_hot_seq_on = get_seqcode(row['Target sgRNA'])
        epi_data_on = get_epi_data(row['Chromosome_on'], row['Start_on'], row['End_on'])
        combined_seq_on = np.concatenate((one_hot_seq_on, epi_data_on), axis=1)

        # Process off-target sequence
        one_hot_seq_off = get_seqcode(row['OT'])
        epi_data_off = get_epi_data(row['Chromosome_off'], row['Start_off'], row['End_off'])
        combined_seq_off = np.concatenate((one_hot_seq_off, epi_data_off), axis=1)

        # Append to respective lists
        combined_on_data.append(combined_seq_on)
        combined_off_data.append(combined_seq_off)

    x_on = np.array(combined_on_data)
    x_off = np.array(combined_off_data)

    x_on = x_on.reshape(-1, 23, 8)
    x_off = x_off.reshape(-1, 23, 8)

    y = merged_df['Cleavage Frequency'].values.reshape(-1, 1)

    if not reg:
        y = np.round(y)

    return (x_on, x_off), y
