import numpy as np
from Bio import SeqIO

def load_data(neg_fasta_filepath: str, pos_fasta_filepath: str) -> tuple[np.ndarray, np.ndarray]:
    _, neg = read_fasta(neg_fasta_filepath)
    _, pos = read_fasta(pos_fasta_filepath)

    x = neg + pos
    y = [0 for i in range(len(neg))] + [1 for i in range(len(pos))]

    index = shuffle_index(len(x))
    return np.asarray(x)[index], np.asarray(y)[index]

def shuffle_index(indexes_number: int) -> np.ndarray:
    index = np.arange(indexes_number)
    np.random.shuffle(index)
    return index

def read_fasta(fasta_filepath: str) -> list[str]:
    ids = []
    seqs = []

    for record in SeqIO.parse(fasta_filepath, "fasta"):
        ids.append(record.id)
        seqs.append(str(record.seq))

    return ids, seqs
