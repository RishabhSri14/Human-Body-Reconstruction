import numpy as np

def positional_encoding(max_len, d_model):
    # Initialize the positional encoding matrix
    pos_enc = np.zeros((max_len, d_model))
    
    # Calculate the positional encoding values for each position and dimension
    for pos in range(max_len):
        for i in range(d_model):
            if i % 2 == 0:
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            else:
                pos_enc[pos, i] = np.cos(pos / (10000 ** ((i - 1) / d_model)))
    
    return pos_enc
