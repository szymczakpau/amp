
STD_AA = list('ACDEFGHIKLMNPQRSTVWY')

def check_if_std_aa(sequence):
    if all(aa in STD_AA for aa in sequence):
        return True
    return False


def check_length(sequence, min_length, max_length):
    if min_length <= len(sequence) <= max_length:
        return True
    return False
