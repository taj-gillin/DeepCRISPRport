import datetime

def debug_print(statements=[], end="\n"):
    ct = datetime.datetime.now()
    print("[", str(ct)[:19], "] ", sep="", end="")
    for statement in statements:
        print(statement, end=" ")
    print(end=end)

def compress_seq(sequence):
    """ 
    Compresses a sequence by converting it to a number.
    Each nucleotide is a number 0-3 (A, C, G, T).
    Each position is a power of 4.
    """
    return sum([4 ** i * "acgt".index(c) for i, c in enumerate(sequence)])

def decompress_seq(number, seq_len=23):
    return "".join(["acgt"[number // 4 ** i % 4] for i in range(seq_len)])
