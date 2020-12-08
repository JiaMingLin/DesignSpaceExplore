from math import ceil
def dsp_cost(n, Pn, Pm):
    return (n**2) * Pn * Pm

def bram_cost(W, C, r, n, m, Tn, Tm, Pn, Pm, precision = 16, double_buffering = True):
    weight = ceil(((r**2) * Pn * Pm * precision)/18) * ceil((ceil(Tn/Pn)*ceil(Tm/Pm)) / 1024)
    IFM = ceil(((n**2) * Pm * precision)/18) * ceil((ceil(W/n) * ceil(Tm/Pm))/1024)
    IFM_temp = ceil((n*m*Pm*precision) / 18) * ceil((ceil(W/n)*ceil(Tm/Pm))/1024)
    OFM = ceil(((m**2)*Pn*32)/18) * ceil((ceil(C/m)*ceil(Tn/Pn))/1024)
    
    print(weight, IFM, IFM_temp, OFM)
    factor = 1
    if double_buffering:
        factor = 2
    
    return factor*(weight + IFM + IFM_temp + OFM)

def latency_tile(W, C, n, m, Tn, Tm, Pn, Pm, datawidth, II=1, precision=16, freq=0.2):
    compute_time = (ceil(C/m) * ceil(Tm / Pm) * ceil(Tn/Pn) * II) * (1/freq) * (10**(-6))
    transfer_time = ceil((n*W*max(Tn,Tm)*precision)/datawidth) * (1/freq) * (10**(-6))
    print("compute_time > transfer_time:", compute_time > transfer_time)
    return max(compute_time, transfer_time)

def latency_init(W, r, n , Tn, Tm, datawidth, precision=16, freq = 16):
    init_time = ceil((Tm*Tn*r*r + n*W*Tm)*precision / datawidth) * (1/freq) * (10**(-6))
    return init_time

def latency_total(W, C, M, N, H, r, n, m, Tn, Tm, Pn, Pm, datawidth, II=1, precision = 16, freq = 0.2):
    total_time = ceil(M/Tm) * ceil(N/Tn) * (ceil(H/m) * \
                        latency_tile(W, C, n, m, Tn, Tm, Pn, Pm, datawidth, II, precision, freq) + \
                        latency_init(W, r, n , Tn, Tm, datawidth, precision))
    return total_time;
