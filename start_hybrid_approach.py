from customICP.normalICP import *
#from customICP.baseICP import *
from learning3d.examples import train_prnet, test_prnet, get_pred_result_prnet

UNIVERSAL_THRESHOLD = 0.001
""" 
structure:
- make prediction of net
- use prediction in ICP
- transform
REPEAT until END
"""


def start_hybrid():
    # read_files()

    prnet_args, prnet, dataloader = get_pred_result_prnet.preprocess()

    A, B = get_pred_result_prnet.predict(prnet_args, prnet, dataloader)

    # prep_ICP

    test_function(get_pred_result_prnet.predict, (prnet_args, prnet, dataloader))

    #icp(A, B, max_iterations=2000000)


if __name__ == "__main__":
    start_hybrid()