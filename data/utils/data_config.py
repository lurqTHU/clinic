feat_cfg = [
            {'type': ['dict'], 'feat_dict': {'男':0, '女': 1}},              # 0. 性别
            {'type': 'threshold', 'threshold': 30.0},                               # 1. 年龄
            {'type': 'bool'},                               # 2. 是否第一步粘附件
            {'type': 'remove'},                               # 3. 是否纳7
            {'type': 'remove'}, #'max_min_norm', 'bound': [0.0, 4.0]},                 # 4. 纳了几个7
            {'type': 'remove'},                                                      # 5. 拔牙与否
            {'type': 'remove'}, #'max_min_norm', 'bound': [0.0, 4.0]},                 # 6. 拔牙个数
            {'type': 'remove'}, #'clip', 'thres': [0.0, 1.0]},                               # 7. 是否橡皮筋牵引
            {'type': 'remove'},                               # 8. 牵引根数
            {'type': 'remove'},#'one_hot', 'num_classes': 5},            # 9. II类牵引2或III类3或颌内1三角4 没有表示为0
            {'type': 'remove'},                            # 10. 前牙区牵引
            {'type': 'remove'},        # 11. 拥挤度分类
            {'type': 'remove'},            # 12. 安氏分类共三类
            {'type': 'remove'},          # 13. 骨性分类共三类
            {'type': 'remove'},                  # 14. 门牙缺失与否
            {'type': 'remove'},                  # 15. 设计步数              
            {'type': 'remove'},                  # 16. 附件牙数
            {'type': 'remove'},                  # 17. 舌侧附件
            {'type': 'max_min_norm', 'bound': [0.0, 18.0]},                  # 18. 优化附件牙数
            {'type': 'nothing'},                  # 19. 优化附件比例%
            {'type': 'max_min_norm', 'bound': [0.0, 4.0]},                  # 20. 上切牙附件个数
            {'type': 'remove'},                  # 21. 有无powerarm
            {'type': 'remove'},                  # 22. PA个数
            {'type': 'remove'},                  # 23. 有无percisioncut
            {'type': 'remove'},                  # 24. PC个数
            {'type': 'remove'},                  # 25. 有无IPR
            {'type': 'remove'},                  # 26. IPR量 
            {'type': 'remove'},                  # 27. 推磨牙向后
            {'type': 'threshold', 'threshold': 1.0}                   # 28. button                     
           ]
  
vas_cfg = [
              {'type': 'remove'},                                              # 基线
              {'type': ['max_idxs', 'threshold'], 
               'index_ranges': [0,1,2,3,4,5,6,7,8], 'threshold': 3.0},                  # 初粘当天
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
               'index_ranges': [0,1,2,3,4,5,6,7,8], 'threshold': 3.0},                  # 初粘当天
              {'type': 'remove'},                  # 初粘后一天
              {'type': 'remove'},                  # day2
              {'type': 'remove'},                  # day3
              {'type': 'remove'},                  # day4
              {'type': 'remove'},                  # day5
              {'type': 'remove'},                  # day6
              {'type': 'remove'}                   # day7
          ]
            
