import numpy as np
import os
from collections import defaultdict
import copy
from data.utils import feat_cfg, vas_cfg, anxiety_cfg


def modify_features(feat, feat_type='feat'):
    backup = copy.deepcopy(feat)
    if feat_type is 'feat':
        args = feat_cfg
    elif feat_type is 'vas':
        args = vas_cfg
    elif feat_type is 'anxiety':
        args = anxiety_cfg
    else:
        raise Exception('Invalid feature type: ', feat_type)
    assert len(feat[0]) == len(args)
    for i in range(len(args)):
         modify_single_feat(feat, i, **args[i], origin=backup)
    check = [[feat[i][j] for i in range(len(feat))] for j in range(len(feat[0]))]
    import pdb;pdb.set_trace()
    return rearange_feat(feat) 


def rearange_feat(feat):
    """
        Flatten and check feat.
    """
    new_feat = []
    for line in feat:
        new_line = []
        for item in line:
            if item is None:
                continue
            elif isinstance(item, list):
                new_line += item
            elif isinstance(item, int) or isinstance(item, float):
                new_line.append(item)
            else:
                raise Exception('Invalid type: ', type(item))
        new_feat.append(new_line)
    return np.array(new_feat, dtype=np.float)


def make_one_hot(num_classes, idx):
    return np.eye(num_classes)[idx, :].tolist()


def modify_single_feat(feat, idx, **args):
    """
        in-place modification
        feat: features, [[feat for item0], [feat for item1], ...]
        idx: index of feature to be modified
        **args: modification configs
    """
    assert args.get('type') is not None
    assert isinstance(feat, list)
    modifications = args.get('type')
    if not isinstance(modifications, list):
        modifications = [modifications]
    for mod in modifications: 
        if mod is 'dict':               # Dict mapping
            assert args.get('feat_dict') is not None
            feat_dict = args['feat_dict']
            for i in range(len(feat)):
                feat[i][idx] = feat_dict[feat[i][idx]]
        elif mod is 'bool':
            for i in range(len(feat)):
                feat[i][idx] = 1 if feat[i][idx]>=1.0 else 0
        elif mod is 'max_min_norm':     # Normalization
            assert args.get('bound') is not None
            minimum, maximum = args['bound']
            for i in range(len(feat)):
                feat[i][idx] = np.float(feat[i][idx] - minimum) / np.float(maximum - minimum)
        elif mod is 'nothing':          # No change
            return
        elif mod is 'one_hot':          # One hot encoding
            assert args.get('num_classes') is not None
            for i in range(len(feat)):
                feat[i][idx] = make_one_hot(args['num_classes'], int(feat[i][idx]))
        elif mod is 'remove':           # Remove
            for i in range(len(feat)):
                feat[i][idx] = None
        elif mod is 'minus_const':      # Minus a constant
            assert args.get('const') is not None
            for i in range(len(feat)):
                feat[i][idx] = feat[i][idx] - args['const']
        elif mod is 'minus_idx':        # Minus the value of another feature index
            assert args.get('index') is not None
            assert args.get('origin') is not None
            origin = args['origin']
            for i in range(len(feat)):
                feat[i][idx] = feat[i][idx] - origin[i][args['index']]
        elif mod is 'clip':   # Clip the value into a fixed interval
            for i in range(len(feat)):
                feat[i][idx] = args['thres'][0] if feat[i][idx] < args['thres'][0] else feat[i][idx]
                feat[i][idx] = args['thres'][1] if feat[i][idx] > args['thres'][1] else feat[i][idx]
        elif mod is 'threshold':            # Mapping to 0 or 1 according to threshold
            assert args.get('threshold') is not None
            thres = args['threshold']
            for i in range(len(feat)):
                feat[i][idx] = 1 if feat[i][idx] >= thres else 0
        elif mod is 'max_idxs':         # Maximum value of specified indexes
            assert args.get('index_ranges') is not None
            index_ranges = args['index_ranges']
            for i in range(len(feat)):
                for j in index_ranges:
                    feat[i][idx] = max([feat[i][idx], feat[i][j]])
        else:
            raise Exception('Invalid modification type: ', args['type'])
