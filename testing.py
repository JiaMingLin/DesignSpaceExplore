from roofline import roofline_model as rm
from workloads.common import *

test_layer = {"nix": 13, "niy": 13, "nif": 192, "nof": 192, "kernel": 3, "type": "conv"}

R=64; C=64; N=128; M=128; Tr=32; Tc=32; Tn=16; Tm=16; S=1; K=3; double_buff = False

print(rm.exec_cycles(R, C, N, M, Tr, Tc, Tn, Tm, S, K, double_buff = double_buff))

board_part = 'v7'
layer_idx = "test_layer"
bits = 32
buswidth = 64
t_factor = 2
save_path = 'files'

print(rm.DSE_layer(test_layer, board_part, layer_idx, bits, buswidth, t_factor,  save_path,\
    double_buff=True, num_hp = 2))