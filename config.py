from yacs.config import CfgNode

_C = CfgNode()

_C.DEVICE = 'cuda'
_C.DEVICE_ID = '0'

# MODEL
_C.NUM_LAYERS = 1 
_C.MID_PLANES = 20
_C.ACTIVATION = 'relu' 
_C.LOSS_TYPE = 'MSE'
_C.USE_ICON = False
_C.TARGET_NAME = 'vas'
_C.SAVE_PREFIX = 'VAS'

# EVALUATE
_C.THRES = 0.1
_C.VAL_METRIC = 'L1'

# SOLVER
_C.BASE_LR = 0.1
_C.WEIGHT_DECAY = 0.0005
_C.BIAS_LR_FACTOR = 1
_C.OPTIMIZER_NAME = 'Adam'

# SCHEDUE
_C.STEPS = [50, 75]
_C.GAMMA = 0.1
_C.MAX_EPOCHS = 20
_C.LOG_PERIOD = 1
_C.EVAL_PERIOD = 1
_C.CHECKPOINT_PERIOD = 5
_C.TRAIN_BATCH_SIZE = 80
_C.VAL_BATCH_SIZE = 30

# DATASET
_C.TRAINVAL_RATIO = 0.7
_C.RANDOM_SEED = 1

# TEST
_C.TEST_FOLDER = './output/'
_C.TEST_PREFIX = 'VAS'

cfg = _C.clone()
