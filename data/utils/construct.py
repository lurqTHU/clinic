import xlrd
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
from data.utils import modify_features, modify_single_feat


sheet_names = ['初粘患者设计', '精调患者设计', 'VAS初粘', 'SAS初粘', 'VAS精调', '焦虑精调']
sheet_flags = ['feat', 'feat', 'vas', 'anxiety', 'vas', 'anxiety']


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
        data = modify_features(data, feat_type=sheet_flags[idx])
        all_feat.append(data)
    for idx in target_idx:
        name = sheet_names[idx]
        keys = Clinic[name]['key']
        data = Clinic[name]['data']
        data = modify_features(data, feat_type=sheet_flags[idx])
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
