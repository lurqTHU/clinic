import numpy as np


def test_params(**args): 
    print(args)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
    param_dict = {'type':1, 'ddd': 2}
    test_params(**param_dict)
