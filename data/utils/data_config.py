feat_cfg = [
            {'type': 'remove'},                                # 0. 性别
            {'type': 'threshold', 'threshold': 40.0},                               # 1. 年龄
            {'type': 'bool'},                               # 2. 是否第一步粘附件
            {'type': 'remove'},                               # 3. 是否纳7
            {'type': 'remove'}, #'max_min_norm', 'bound': [0.0, 4.0]},                 # 4. 纳了几个7
            {'type': 'bool'},                                                      # 5. 拔牙与否
            {'type': 'max_min_norm', 'bound': [0.0, 4.0]},                 # 6. 拔牙个数
            {'type': 'remove'}, #'clip', 'thres': [0.0, 1.0]},                               # 7. 是否橡皮筋牵引
            {'type': 'remove'},                               # 8. 牵引根数
            {'type': 'remove'},#'one_hot', 'num_classes': 5},            # 9. II类牵引2或III类3或颌内1三角4 没有表示为0
            {'type': 'remove'},                            # 10. 前牙区牵引
            {'type': 'dict', 'feat_dict': {0:0.0, 1:float(1)/3, 2: float(2)/3, 3:1.0, 4:0.0}},        # 11. 拥挤度分类
            {'type': 'remove'},            # 12. 安氏分类共三类
            {'type': 'remove'},          # 13. 骨性分类共三类
            {'type': 'remove'},                  # 14. 门牙缺失与否
            {'type': 'remove'},                  # 15. 设计步数              
            {'type': 'remove'},                  # 16. 附件牙数
            {'type': 'threshold', 'threshold': 1.0},                  # 17. 舌侧附件
            {'type': 'max_min_norm', 'bound': [0.0, 18.0]},                  # 18. 优化附件牙数
            {'type': 'nothing'},                  # 19. 优化附件比例%
            {'type': 'max_min_norm', 'bound': [0.0, 4.0]},                  # 20. 上切牙附件个数
            {'type': 'remove'},                  # 21. 有无powerarm
            {'type': 'remove'},                  # 22. PA个数
            {'type': 'bool'},                  # 23. 有无percisioncut
            {'type': 'remove'},                  # 24. PC个数
            {'type': 'bool'},                  # 25. 有无IPR
            {'type': 'remove'},                  # 26. IPR量 
            {'type': 'bool'},                  # 27. 推磨牙向后
            {'type': ['clip', 'max_min_norm'], 'thres': [0.0, 4.0], 'bound': [0.0, 4.0]}                   # 28. button                     
           ]
  
vas_cfg = [
              {'type': 'remove'},                                              # 基线
              {'type': ['max_idxs', 'threshold'], 
               'index_ranges': [0,1,2,3,4,5,6,7,8], 'threshold': 4.0},                  # 初粘当天
              {'type': 'remove'},                  # 初粘后一天
              {'type': 'remove'},                  # day2
              {'type': 'remove'},                  # day3
              {'type': 'remove'},                  # day4
              {'type': 'remove'},                  # day5
              {'type': 'remove'},                  # day6
              {'type': 'remove'}                   # day7
             ]

anxiety_cfg =  [
              {'type': 'remove'},                                              # 基线
              {'type': ['max_idxs', 'threshold'], 
               'index_ranges': [0,1,2,3,4,5,6,7,8], 'threshold': 45.0},                  # 初粘当天
              {'type': 'remove'},                  # 初粘后一天
              {'type': 'remove'},                  # day2
              {'type': 'remove'},                  # day3
              {'type': 'remove'},                  # day4
              {'type': 'remove'},                  # day5
              {'type': 'remove'},                  # day6
              {'type': 'remove'}                   # day7
             ]

qol_cfg = [
              {'type': 'remove'},                                              # 基线
              {'type': ['max_idxs', 'threshold'], 
               'index_ranges': [0,1,2,3,4,5,6,7,8], 'threshold': 20.0},                  # 初粘当天
              {'type': 'remove'},                  # 初粘后一天
              {'type': 'remove'},                  # day2
              {'type': 'remove'},                  # day3
              {'type': 'remove'},                  # day4
              {'type': 'remove'},                  # day5
              {'type': 'remove'},                  # day6
              {'type': 'remove'}                   # day7
          ]
           
icon_cfg = [
              {'type': 'remove'},      # 美观检查
              {'type': 'remove'},      # 反合 5
              {'type': 'remove'},      # 前牙垂直关系 4
              {'type': 'remove'},      # 上牙拥挤/间隙5
              {'type': 'remove'},      # 失状向关系3
              {'type': 'max_min_norm', 'bound': [0.0, 86.0]}  # ICON
           ] 
