import xlrd
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

source_keys = ['初粘患者设计', '精调患者设计', 'VAS初粘', '焦虑初粘', 'VAS精调', '焦虑精调']


def parse_excel(excel_path, output_path):
    """
        Clinic: {'keys':[[usr0], [usr1], ... 
                        ]
                }
    """

    data = xlrd.open_workbook(excel_path)

    Clinic = defaultdict(dict)
    for i, key in enumerate(source_keys):
        tbl = data.sheet_by_name(key)
        feat = []
        # The first row contains keys
        row = tbl.row(0)
        keys = [row[idx].value for idx in range(1, len(row))] 
        # Skip the first row
        for j in range(1, tbl.nrows):
            row = tbl.row(j)
            name = row[0].value
            if name != '':
                feat.append([row[idx].value for idx in range(1, len(row))])
        Clinic[key]['key'] = keys
        Clinic[key]['data'] = feat 
        
   
#    with open(output_path, 'wb') as f:
#        json.dump(Clinic, f)

    return Clinic
    

def analyze_feat(table, plot_dir): 
    index = [0, 1]
    etp = []
    for idx in index:
        key = source_keys[idx]
        keys = table[key]['key']
        data = table[key]['data']
        import pdb;pdb.set_trace()
        data = np.array(data, dtype=np.float)
        for i, k in enumerate(keys):
            feat = np.array(data[:,i])
            hist, values = np.histogram(feat) 
            entropy = cal_entropy(feat)
            etp.append(entropy)
    import pdb;pdb.set_trace()
 

def modify_single_feat(feat, idx, **args):
    """
        Modify in-place
        feat: features, [[item0], [item1], ...]
        idx: feature dim to be modified
    """
    assert args.get('type') is not None
    if args['type'] is 'dict':
        assert args.get('feat_dict') is not None
        feat_dict = args['feat_dict']
        for i in range(len(feat)):
            feat[i][idx] = feat_dict[feat[i][idx]]
    if args['type'] is 'max_min_norm':
        assert args.get('bound') is not None
        minimum, maximum = args['bound']
        for i in range(len(feat)):
            feat[i][idx] = np.float(feat[i][idx] - mininum) / np.float(maximum - minimum)
    if args['type'] is 'nothing':
        return
    return
   

def modify_features(feat):
    args = [
            {'type': 'dict', 'feat_dict': {'男':0, '女': 1},  # 性别
             'type': 'nothing',                               # 年龄
             'type': 'nothing',                               # 是否第一步粘附件
             'type': 'nothing',                               # 是否纳7
             'type': 'nothing',                               # 纳了几个7
             'type': 'nothing',                               # 拔牙与否
             'type': 'nothing',                               # 拔牙个数
             'type': 'nothing',                               # 是否橡皮筋牵引
             'type': 'nothing',                               # 牵引根数
             'type': 'nothing',                  # II类牵引2或III类3或颌内1 没有表示为0
             'type': 'nothing',                  # 前牙区牵引
             'type': 'nothing',                  # 拥挤度分类
             'type': 'nothing',                  # 安氏分类共三类
             'type': 'nothing',                  # 骨性分类共三类
             'type': 'nothing',                  # 门牙缺失与否
             'type': 'nothing',                  # 设计步数              
             'type': 'nothing',                  # 附件牙数
             'type': 'nothing',                  # 舌侧附件
             'type': 'nothing',                  # 优化附件牙数
             'type': 'nothing',                  # 优化附件比例%
             'type': 'nothing',                  # 上切牙附件个数
             'type': 'nothing',                  # 有无powerarm
             'type': 'nothing',                  # PA个数
             'type': 'nothing',                  # 有无percisioncut
             'type': 'nothing',                  # PC个数
             'type': 'nothing',                  # 有无IPR
             'type': 'nothing',                  # IPR量 
             'type': 'nothing',                  # 推磨牙向后
             'type': 'nothing'                   # button                
             }     
           ]
    return  


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
    json_path = '/space1/home/lurq/code/clinic/dataset/data.json'
    img_dir = '/space1/home/lurq/code/clinic/dataset' 
    data = parse_excel(file_path, json_path)
    analyze_feat(data, img_dir)
