import xlrd
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
from dataset.modify_features import modify_features, modify_single_feat


sheet_names = ['初粘患者设计', '精调患者设计', 
               'VAS初粘', 'SAS初粘', 'QoL初粘' , 
               'VAS精调', '焦虑精调', 'QoL精调',
               'ICON初粘', 'ICON精调']
sheet_flags = ['feat', 'feat', 
               'vas', 'anxiety', 'qol', 
               'vas', 'anxiety', 'qol',
               'icon', 'icon']


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
    

def construct_Clinic(excel_path, target_name, use_icon):
    Clinic = parse_excel(excel_path)
    # Select sheet content
    feat_idx = [0, 1]
    target_idx_dict = {'vas': [2, 5], 'sas': [3, 6], 'qol': [4, 7]}
    target_idx = target_idx_dict[target_name]
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
    if use_icon:
        # Add ICON feature
        icon_idx = [8, 9]
        all_icon = []
        for idx in icon_idx:
            name = sheet_names[idx]
            keys = Clinic[name]['key']
            data = Clinic[name]['data']
            data = modify_features(data, feat_type=sheet_flags[idx])
            all_icon.append(data)    
        all_icon = np.concatenate(all_icon, axis = 0)
        all_feat = np.concatenate((all_feat, all_icon), axis = 1)
    return {'feat': all_feat, 'target': all_target}


def partition_dataset(data_path, target_name='vas', 
                      use_icon=True, ratio=0.8, seed=0):
    np.random.seed(seed)

    infos = construct_Clinic(data_path, target_name, use_icon)
    feats = infos['feat']
    targets = infos['target']

    print('Feature dimension: ', feats.shape[1])
           
    # Split dataset into trainval and test set
    positive_mask = np.where(targets == 1)[0]
    negative_mask = np.where(targets == 0)[0] 
    trainval_mask = []
    test_mask = []
    for pick in (positive_mask, negative_mask):
        total = len(pick)
        rand_uniform = np.random.uniform(0, 1, total)
        trainval_mask.extend(pick[rand_uniform < ratio])
        test_mask.extend(pick[rand_uniform >= ratio])
    # Further split trainval set into train and val set
    train_mask = []
    val_mask = []
    rand_uniform = np.random.uniform(0, 1, len(trainval_mask))
    train_mask.extend(np.array(trainval_mask)[rand_uniform<0.75])
    val_mask.extend(np.array(trainval_mask)[rand_uniform>=0.75])
      
    print('Trainval mask:', trainval_mask)
    print('test mask:', test_mask)
    print('{:>23}{:>10}{:>7}{:>5}{:>6}'\
          .format('Total', 'Trainval', 'Train', 'Val', 'Test'))
    print('Negative counts: {:>6}{:>10}{:>7}{:>5}{:>6}'\
          .format(np.sum(targets==0), 
                  np.sum(targets[trainval_mask]==0),
                  np.sum(targets[train_mask]==0),
                  np.sum(targets[val_mask]==0),
                  np.sum(targets[test_mask]==0)))
    print('Positive counts: {:>6}{:>10}{:>7}{:>5}{:>6}'\
          .format(np.sum(targets==1), 
                  np.sum(targets[trainval_mask]==1),
                  np.sum(targets[train_mask]==1),
                  np.sum(targets[val_mask]==1),
                  np.sum(targets[test_mask]==1)))
    return feats, targets, (trainval_mask, \
           train_mask, val_mask, test_mask)


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
