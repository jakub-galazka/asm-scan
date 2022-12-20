import numpy as np
from Bio import SeqIO

def load_data(neg_filepath: str, pos_filepath: str) -> tuple[np.ndarray, np.ndarray]:
    neg = read_fasta(neg_filepath)
    pos = read_fasta(pos_filepath)

    x = neg + pos
    y = [0 for i in range(len(neg))] + [1 for i in range(len(pos))]

    index = shuffle_index(len(x))
    return np.asarray(x)[index], np.asarray(y)[index]

def shuffle_index(indexes_number: int) -> np.ndarray:
    index = np.arange(indexes_number)
    np.random.shuffle(index)
    return index

def read_fasta(filepath: str) -> list[str]:
    return [str(record.seq) for record in SeqIO.parse(filepath, "fasta")]
