import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
import statistics

sys.path.append('workloads/')

from tqdm import trange
from math import ceil
from common import *

def bram_usage(beta_in, beta_wgt, beta_out, Tn, Tm, bits = 8, double_buff=False):
    
    usage_in = ceil(beta_in/(1024.*Tn))*ceil(Tn*bits/18.)
    usage_wgt = ceil(beta_wgt/(1024.*Tn*Tm))*ceil(Tn*Tm*bits/18.)
    usage_out = ceil(beta_out/(1024.*Tm))*ceil(Tm*32/18.)
    
    if double_buff:
        return 2*(usage_in + usage_wgt + usage_out)
    else:
        return usage_in + usage_wgt + usage_out

def dsp_usage(Tn, Tm, bits):
    dsp_factor = 1
    if bits is 8:
        dsp_factor = 2
    elif bits is 16:
        dsp_factor = 1
    elif bits is 32:
        dsp_factor = 0.2
    else:
        dsp_factor = 1
    
    return ceil((Tn)*Tm / dsp_factor)

def compute_bound(num_dsp, freq, bits = 8):
    
    if bits is 8:
        dsp_factor = 2
    elif bits is 16:
        dsp_factor = 1
    elif bits is 32:
        dsp_factor = 0.2
    else:
        dsp_factor = 1
        
    return 2 * num_dsp * dsp_factor * freq

# num_hp: number of high performance ports
def bandwidth_roof(buswidth, freq, num_hp = 1):
    return (buswidth/8) * num_hp * freq
    

def ctc_ratio(R, C, N, M, Tr, Tc, Tn, Tm, S, K, BRAM, DSP, bits = 8, double_buff = False):
    beta_in = Tn * (S*Tr + K-S) * (S*Tc + K-S)
    beta_wgt = Tn * Tm * K * K
    beta_out = Tm * Tr * Tc
    
    alpha_in = ceil(M/Tm) * ceil(N/Tn) * ceil(R/Tr) * ceil(C/Tc)
    alpha_wgt = ceil(M/Tm) * ceil(N/Tn) * ceil(R/Tr) * ceil(C/Tc)
    alpha_out = ceil(M/Tm) * ceil(R/Tr) * ceil(C/Tc)
    
    bram_cost = bram_usage(beta_in, beta_wgt, beta_out, Tn, Tm, bits, double_buff)
    dsp_cost = dsp_usage(Tn, Tm, bits)

    if bram_cost > BRAM:
        return -1, bram_cost, dsp_cost
    
    if dsp_cost > DSP:
        return -1, bram_cost, dsp_cost
    
    num_ops = 2 * R * C * M * N * K * K
    num_ext_access = alpha_in * beta_in + alpha_wgt * beta_wgt + alpha_out * beta_out
    return num_ops/float(num_ext_access), bram_cost, dsp_cost

def exec_cycles(R, C, N, M, Tr, Tc, Tn, Tm, S, K, double_buff = False):

    # buffer size for IFM, WGT, PSUM
    beta_in = Tn * (S*Tr + K-S) * (S*Tc + K-S)
    beta_wgt = Tn * Tm * K * K
    beta_out = Tm * Tr * Tc
    
    alpha_in = ceil(M/Tm) * ceil(N/Tn) * ceil(R/Tr) * ceil(C/Tc)
    alpha_wgt = ceil(M/Tm) * ceil(N/Tn) * ceil(R/Tr) * ceil(C/Tc)
    alpha_out = ceil(M/Tm) * ceil(R/Tr) * ceil(C/Tc)

    conv_cycle = ceil(M/Tm)*ceil(N/Tn)*ceil(R/Tr)*ceil(C/Tc)*(Tr*Tc*K*K)

    if double_buff:
        return ceil(M/Tm)*ceil(N/Tn)*ceil(R/Tr)*ceil(C/Tc)*(Tr*Tc*K*K)
    else:
        read_ifm_cycle = beta_in/(Tn)
        read_wgt_cycle = beta_wgt/(Tn)
        write_output_cycle = beta_out/(Tm)
        # print("read_ifm_cycle = {}, read_wgt_cycle = {}, conv_cycle = {}, write_output_cycle = {}".format(\
            # read_ifm_cycle, read_wgt_cycle, Tr*Tc*K*K, write_output_cycle))
        # print("Tr round = {}, Tc round={}, Tm round = {}".format(ceil(R/Tr), ceil(C/Tc), ceil(M/Tm)))
        single_tile_cycle = write_output_cycle+ceil(N/Tn)*(Tr*Tc*K*K+read_ifm_cycle+read_wgt_cycle)
        return ceil(M/Tm)*ceil(R/Tr)*ceil(C/Tc)*single_tile_cycle


def compute_roof(R, C, N, M, Tr, Tc, Tn, Tm, S, K, num_dsp, freq, bits=8, double_buff = False):
    num_ops = 2 * R * C * M * N * K * K
    num_exec_cycles = exec_cycles(R, C, N, M, Tr, Tc, Tn, Tm, S, K, double_buff)
    
    bound = compute_bound(num_dsp, freq, bits)
    
    return min((num_ops*freq)/float(num_exec_cycles), bound)

# tiling size increased by tiling_factor
def tiling_candidates(R, C, N, M, tiling_factor=4):
    if (R or C) == 1:
        ls_Tr = np.arange(0, 1, 1); ls_Tr[0]+=1
        ls_Tc = np.arange(0, 1, 1); ls_Tc[0]+=1
    elif (R or C) == tiling_factor:
        ls_Tr = np.arange(0, R, tiling_factor); ls_Tr[0]+=tiling_factor
        ls_Tc = np.arange(0, C, tiling_factor); ls_Tc[0]+=tiling_factor
    else:
        ls_Tr = np.arange(0, R, tiling_factor); ls_Tr = ls_Tr+tiling_factor
        ls_Tc = np.arange(0, C, tiling_factor); ls_Tc = ls_Tc+tiling_factor

    ls_Tn = np.arange(0,N,tiling_factor)
    if (N < tiling_factor):
        ls_Tn += N
    else:
        ls_Tn += tiling_factor
    
    ls_Tm = np.arange(0,M,tiling_factor)
    if (M < tiling_factor):
        ls_Tm += M
    else:
        ls_Tm += tiling_factor
        
    a = [\
        [int(x) for x in ls_Tr],\
        [int(x) for x in ls_Tc],\
        [int(x) for x in ls_Tn],\
        [int(x) for x in ls_Tm]
    ]


    return list(itertools.product(*a))

def RF_Model(layer_meta, board, \
    tiling_factor = 4, bits = 8, l_type = 'conv', buswidth = 64, double_buff = False, num_hp = 1):
    R = layer_meta['niy']
    C = layer_meta['nix']
    N = layer_meta['nif']
    M = layer_meta['nof']
    K = layer_meta['kernel']
    S = layer_meta['stride'] if 'stride' in layer_meta.keys() else 1 
    BRAM = board['bram']
    num_dsp = board['dsp']
    freq = board['freq']
    
    comp_bnd = compute_bound(num_dsp, freq, bits) # platform computation bound
    bw_bnd = bandwidth_roof(buswidth, freq, num_hp = num_hp)
    
    if l_type is 'fc':
        N = R*C*N
        R = 1; C = 1
    
    tiling_params = tiling_candidates(R, C, N, M, tiling_factor)
    
    pair = []
    params = []
    costs = []
    for i in trange(len(tiling_params)):
        (Tr, Tc, Tn, Tm) = tiling_params[i]
        if Tm < Tn:
            continue
        ctc, bram_cost, dsp_cost = ctc_ratio(R, C, N, M, Tr, Tc, Tn, Tm, S, K, BRAM, num_dsp, bits, double_buff)
        attainable = compute_roof(R, C, N, M, Tr, Tc, Tn, Tm, S, K, num_dsp, freq, bits, double_buff)
        
        if ctc > 0:
            if ctc < comp_bnd/float(bw_bnd):
                attainable = min(attainable, ctc*bw_bnd)
            pair.append((float(ctc), float(attainable)))
            params.append((Tr, Tc, Tn, Tm))
            costs.append((bram_cost, dsp_cost))
    return pair, params, costs, comp_bnd, bw_bnd

def DSE_layer(layer_meta, board_part, layer_idx, bits, buswidth, t_factor,  save_path,\
    double_buff=False, num_hp = 1):
    
    fpga = fpga_boards[board_part]

    (pair_ls, params, costs, comp_bnd, bw_bnd) = RF_Model(\
        layer_meta, fpga, \
        tiling_factor = t_factor, bits=bits, l_type = layer_meta['type'], \
        buswidth = buswidth, double_buff= double_buff, num_hp = num_hp)
    # pair_ls = (ctc, attainable)
    max_throughput_value = max(pair_ls, key = lambda it: it[1])[1]
    max_throughput_ls = list(filter(lambda it: it[0][1]==max_throughput_value, list(zip(pair_ls, params, costs))))
    solution = max(max_throughput_ls, key = lambda it: it[0][0])
    save_fig_name = save_path+'/{}_{}_tf_{}_bits_{}_bus_{}.png'.format(board_part, layer_idx, t_factor, bits, buswidth)
       
    plot_roofline(pair_ls, comp_bnd, bw_bnd, solution, save_fig_name)
    
    return pair_ls, comp_bnd, bw_bnd, solution

def plot_roofline(pair_ls, comp_bnd, bw_bnd, solution, save_fig_name):

    # Create data
    unzipped_object = zip(*pair_ls)
    unzipped_list = list(unzipped_object)
    ctc = np.array(unzipped_list[0]); max_ctc = max(ctc)
    comp = np.array(unzipped_list[1]); 
    area = np.pi

    intersect = comp_bnd / float(bw_bnd)
    x = (0, intersect, max_ctc+10)
    y = (0, comp_bnd, comp_bnd)

    # Plot roof line
    plt.scatter(ctc, comp, s=area, c='green', alpha=0.5)
    plt.plot(x, y, 'ro-')

    # plot the solution
    sol_txt = str(solution[1])
    plt.scatter(solution[0][0], solution[0][1], marker='^', s = 100)
    plt.annotate(sol_txt, solution[0])

    plt.title('Roof Line Model')
    plt.xlabel('CTC Ratio')
    plt.ylabel('Attainable Performance')

    plt.grid()
    plt.savefig(save_fig_name)
    plt.close()