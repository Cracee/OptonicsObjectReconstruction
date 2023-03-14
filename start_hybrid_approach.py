from customICP.icp import do_classic_icp
from customICP.hybrid_helper import *
from learning3d.examples import train_prnet, test_prnet, get_pred_result_prnet


do_classic_icp()
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

    # load_prnet()

    # prep_ICP

    while t > UNIVERSAL_THRESHOLD:
        pred_matrix = ask_prnet()

        ICP_step()

        transform()

    result()

if __name__ == "__main__":
    start_hybrid()