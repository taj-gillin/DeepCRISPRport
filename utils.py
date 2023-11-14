import datetime
import numpy as np
from functools import reduce

# sgRNA Encoding
ntmap = {'A': (1, 0, 0, 0), 'C': (0, 1, 0, 0), 'G': (0, 0, 1, 0), 'T': (0, 0, 0, 1)}
def get_seqcode(seq):
    return np.array(reduce(lambda x, y: x + y, [ntmap[c] for c in seq.upper()])).reshape(1, len(seq), -1)

def get_epi_data(chromosome, start, end):
    return np.zeros((1, end - start + 1, 4))

# Printing
def debug_print(statements=[], end="\n"):
    ct = datetime.datetime.now()
    print("[", str(ct)[:19], "] ", sep="", end="")
    for statement in statements:
        print(statement, end=" ")
    print(end=end)
