import xlrd
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
from data.data_config import feat_cfg, target_cfg
import copy


sheet_names = ['初粘患者设计', '精调患者设计', 'VAS初粘', '焦虑初粘', 'VAS精调', '焦虑精调']
sheet_flags = ['feat', 'feat', 'target', 'target', 'target', 'target']


def parse_excel(excel_path):
    """
        Clinic: {'sheet_name0':{                                     
                               'key': []
                               'data': [[usr0], [usr1], ... ]
                               }
                 'sheet_name1':{ 
                                ...
                               }
                }
    """

    data = xlrd.open_workbook(excel_path)

    Clinic = defaultdict(dict)
    for i, name in enumerate(sheet_names):
        tbl = data.sheet_by_name(name)
        feat = []
        # The first row contains keys, the first column is the name of user
        row = tbl.row(0)
        keys = [row[idx].value for idx in range(1, len(row))] 
        # Skip the first row
        for j in range(1, tbl.nrows):
            row = tbl.row(j)
            user_name = row[0].value
            if user_name != '':
                feat.append([row[idx].value for idx in range(1, len(row))])
        Clinic[name]['key'] = keys
        Clinic[name]['data'] = feat 

    return Clinic
    

def construct_Clinic(excel_path):
    Clinic = parse_excel(excel_path)
    feat_idx = [0, 1]
    target_idx = [2, 4]
    all_feat = []
    all_target = []
    for idx in feat_idx:
        name = sheet_names[idx]
        keys = Clinic[name]['key']
        data = Clinic[name]['data']
        data = modify_features(data, feat_type='feat')
        all_feat.append(data)
    for idx in target_idx:
        name = sheet_names[idx]
        keys = Clinic[name]['key']
        data = Clinic[name]['data']
        data = modify_features(data, feat_type='target')
        all_target.append(data)
    all_feat = np.concatenate(all_feat, axis = 0)
    all_target = np.concatenate(all_target, axis = 0)
    return {'feat': all_feat, 'target': all_target}


def analyze_feat(table, plot_dir, plot):
    """
        Calculate entropy and plot histogram for each feature index.
        table: Clinic dataset
        plot_dir: save dir for plots
        plot: True, False
    """ 
    index = [0, 1]
    for idx in index:
        key = sheet_names[idx]
        keys = table[key]['key']
        data = table[key]['data']
        # Modify the sex str to int
        modify_single_feat(data, 0, type='dict', feat_dict={'男':0, '女':1})
        data = np.array(data, dtype=np.float)
        for i, k in enumerate(keys):
            feat = np.array(data[:,i])
            entropy = cal_entropy(feat)
            if plot:
                save_path = os.path.join(plot_dir, key+'_'+k)
                fig = plt.figure()
                plt.hist(feat)
                plt.title(k+': %.3f'% entropy)
                plt.savefig(save_path)
                plt.close()


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
        if mod is 'dict':
            assert args.get('feat_dict') is not None
            feat_dict = args['feat_dict']
            for i in range(len(feat)):
                feat[i][idx] = feat_dict[feat[i][idx]]
        elif mod is 'max_min_norm':
            assert args.get('bound') is not None
            minimum, maximum = args['bound']
            for i in range(len(feat)):
                feat[i][idx] = np.float(feat[i][idx] - minimum) / np.float(maximum - minimum)
        elif mod is 'nothing':
            return
        elif mod is 'one_hot':
            assert args.get('num_classes') is not None
            for i in range(len(feat)):
                feat[i][idx] = make_one_hot(args['num_classes'], int(feat[i][idx]))
        elif mod is 'remove':
            for i in range(len(feat)):
                feat[i][idx] = None
        elif mod is 'minus_const':
            assert args.get('const') is not None
            for i in range(len(feat)):
                feat[i][idx] = feat[i][idx] - args['const']
        elif mod is 'minus_idx':
            assert args.get('index') is not None
            assert args.get('origin') is not None
            origin = args['origin']
            for i in range(len(feat)):
                feat[i][idx] = feat[i][idx] - origin[i][args['index']]
        elif mod is 'clip':
            for i in range(len(feat)):
                feat[i][idx] = args['thres'][0] if feat[i][idx] < args['thres'][0] else feat[i][idx]
                feat[i][idx] = args['thres'][1] if feat[i][idx] > args['thres'][1] else feat[i][idx]
        else:
            raise Exception('Invalid modification type: ', args['type'])

  
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


def modify_features(feat, feat_type='feat'):
    backup = copy.deepcopy(feat)
    if feat_type is 'feat':
        args = feat_cfg
    elif feat_type is 'target':
        args = target_cfg
    else:
        raise Exception('Invalid feature type: ', feat_type)
    assert len(feat[0]) == len(args)
    for i in range(len(args)):
         modify_single_feat(feat, i, **args[i], origin=backup)
    check = [[feat[i][j] for i in range(len(feat))] for j in range(len(feat[0]))]
    return rearange_feat(feat) 


def cal_entropy(data):
    total = data.size
    values = np.unique(data)
    prob = np.zeros(len(values), dtype=np.float)
    for i, v in enumerate(values):
        cnt = np.sum(data == v)
        prob[i] = np.float(cnt) / np.float(total)
    entropy = np.sum(-np.log2(prob) * prob)
    return entropy


if __name__ == '__main__':
    file_path = '/space1/home/lurq/code/clinic/dataset/raw.xlsx'
    plot_dir = '/space1/home/lurq/code/clinic/dataset' 

#    data = parse_excel(file_path)
#    analyze_feat(data, plot_dir, plot=False)

    construct_Clinic(file_path)
