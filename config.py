SEED = 2021
PAD = 0
START = 1
TOTAL_EID = 13523
TOTAL_CAT = 10000
TOTAL_PART = 7
TOTAL_RESP = 2
TOTAL_ETIME = 300
TOTAL_LTIME_S = 300
TOTAL_LTIME_M = 1440
TOTAL_LTIME_D = 365


DEVICE = [0]                                # [0]: use cuda:0 to train
MAX_SEQ = 100                               # The maximum length of inputting sequence at a time
MIN_SEQ = 2                                 # The minimal length of inputting sequence at a time
OVERLAP_SEQ = 60                            # Split a training sample every OVERLAP_SEQ words
MODEL_DIMS = 256                            # Model dimension
FEEDFORWARD_DIMS = 256                      # Hidden dimension of FFN
N_HEADS = 8                                 # Number of Attention heads
NUM_ENCODER = NUM_DECODER = 2               # Number of Encoder/Decoder in Transformer
DROPOUT = 0.1                               # Dropout rate
BATCH_SIZE = 256                            # Batch size
LEARNING_RATE = 5e-4                        # Base learning rate

EPOCH = 100                                 # Maximum number of training
D_PIV = 32