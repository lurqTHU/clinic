feat_cfg = [
            {'type': ['dict', 'one_hot'], 'feat_dict': {'男':0, '女': 1}, 'num_classes': 2},  # 性别
            {'type': 'max_min_norm', 'bound': [15.0, 50.0]},                               # 年龄
            {'type': 'nothing'},                               # 是否第一步粘附件
            {'type': 'remove'},                               # 是否纳7
            {'type': 'remove'},                               # 纳了几个7
            {'type': 'nothing'},                               # 拔牙与否
            {'type': 'max_min_norm', 'bound': [0.0, 4.0]},                               # 拔牙个数
            {'type': 'clip', 'thres': [0.0, 1.0]},                               # 是否橡皮筋牵引
            {'type': 'max_min_norm', 'bound': [0.0, 4.0]},                               # 牵引根数
            {'type': 'one_hot', 'num_classes': 5},                  # II类牵引2或III类3或颌内1三角4 没有表示为0
            {'type': 'remove'},                  # 前牙区牵引
            {'type': 'one_hot', 'num_classes': 5},                  # 拥挤度分类
            {'type': ['minus_const', 'one_hot'], 'const': 1, 'num_classes': 3},                  # 安氏分类共三类
            {'type': ['minus_const', 'one_hot'], 'const': 1, 'num_classes': 3},                  # 骨性分类共三类
            {'type': 'nothing'},                  # 门牙缺失与否
            {'type': 'max_min_norm', 'bound': [10.0, 75.0]},                  # 设计步数              
            {'type': 'max_min_norm', 'bound': [10.0, 25.0]},                  # 附件牙数
            {'type': 'max_min_norm', 'bound': [0.0, 4.0]},                  # 舌侧附件
            {'type': 'max_min_norm', 'bound': [0.0, 18.0]},                  # 优化附件牙数
            {'type': 'nothing'},                  # 优化附件比例%
            {'type': 'max_min_norm', 'bound': [0.0, 4.0]},                  # 上切牙附件个数
            {'type': 'nothing'},                  # 有无powerarm
            {'type': 'remove'},                  # PA个数
            {'type': 'nothing'},                  # 有无percisioncut
            {'type': 'max_min_norm', 'bound': [0.0, 4.0]},                  # PC个数
            {'type': 'nothing'},                  # 有无IPR
            {'type': 'max_min_norm', 'bound': [0.0, 4.0]},                  # IPR量 
            {'type': 'nothing'},                  # 推磨牙向后
            {'type': 'max_min_norm', 'bound': [0.0, 4.0]}                   # button                     
           ]
  
target_cfg = [
              {'type': 'remove'},                                              # 基线
              {'type': ['minus_idx', 'max_min_norm'], 'index': 0, 'bound': [0.0, 10.0]},                  # 初粘当天
              {'type': 'remove'},                  # 初粘后一天
              {'type': 'remove'},                  # day2
              {'type': 'remove'},                  # day3
              {'type': 'remove'},                  # day4
              {'type': 'remove'},                  # day5
              {'type': 'remove'},                  # day6
              {'type': 'remove'}                   # day7
             ]
              
