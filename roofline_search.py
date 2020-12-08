from workloads import vgg16, simple_net, resnet50, resnet18
from workloads.common import *

from roofline.roofline_model import *

# net = vgg16.network(resolutions['original'])
# save_path = 'files/vgg16'

net = resnet18.network(resolutions['original'])
save_path = 'files/resnet18'

# net = simple_net.network(resolutions['original'])
# save_path = 'files/simple_net'

board_part = 'zcu102'
double_buff=True
num_hp = 1
t_factor = 2
bits = 8
buswidth = 128

sol_ls = []
cost_ls = []
layer_cycles = []
for layer_idx in net.keys():
    print("cuttent layer: ", layer_idx)
    layer_meta = net[layer_idx]
    pair_ls, comp_bnd, bw_bnd, solution = DSE_layer(\
        layer_meta, board_part, layer_idx, bits, buswidth, t_factor, save_path, \
        double_buff=double_buff, num_hp = num_hp)

    sol_ls.append(solution[1])
    R = layer_meta['niy']; C = layer_meta['nix']; N = layer_meta['nif']; M = layer_meta['nof']; K = layer_meta['kernel']
    S = 1
    Tr = solution[1][0];Tc = solution[1][1];Tn = solution[1][2];Tm = solution[1][3]
    layer_cycles.append(exec_cycles(R, C, N, M, Tr, Tc, Tn, Tm, S, K))

# averaged to final solution
avg_Tr = list(zip(*sol_ls))[0]; avg_Tr = statistics.mean(avg_Tr)
avg_Tc = list(zip(*sol_ls))[1]; avg_Tc = statistics.mean(avg_Tc)
avg_Tn = list(zip(*sol_ls))[2]; avg_Tn = statistics.mean(avg_Tn)
avg_Tm = list(zip(*sol_ls))[3]; avg_Tm = statistics.mean(avg_Tm)

MAX_K=7; MAX_S = 2
beta_in = avg_Tn * (MAX_S*avg_Tr + MAX_K-MAX_S) * (MAX_S*avg_Tc + MAX_K-MAX_S)
beta_wgt = avg_Tn * avg_Tm * MAX_K * MAX_K
beta_out = avg_Tm * avg_Tr * avg_Tc

bram_cost = bram_usage(beta_in, beta_wgt, beta_out, avg_Tn, avg_Tm, bits = bits, double_buff=double_buff)
dsp_cost = dsp_usage(avg_Tn, avg_Tm, bits)

all_layer_cycle = 0
for layer_idx in net.keys():
    layer_meta = net[layer_idx]
    # count each layer time
    R = layer_meta['niy']; C = layer_meta['nix']; N = layer_meta['nif']; M = layer_meta['nof']; K = layer_meta['kernel']
    S = 1
    all_layer_cycle += exec_cycles(R, C, N, M, avg_Tr, avg_Tc, avg_Tn, avg_Tm, S, K)

# print(sol_ls)
print("averaged Tr = {}, Tc = {}, Tn = {}, Tm = {}".format(avg_Tr, avg_Tc, avg_Tn, avg_Tm))
print("sum of each layer exec_cycles: ", sum(layer_cycles))
print("sum of uniformed layer exec_cycles: ", all_layer_cycle)
print("average latency(ms): ", (all_layer_cycle/(200*(10**6)))*(10**3))
print("optimal latency(ms): ", (sum(layer_cycles)/(200*(10**6)))*(10**3))
print("resuorce cost, BRAM = {}, DSP = {}".format(bram_cost, dsp_cost))
