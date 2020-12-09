resolutions={
    "224":{"nix": 224, "niy":224},
    "original":{"nix": 256, "niy":256},
    "720p":{"nix": 1280, "niy":720},
    "1080p":{"nix": 1920, "niy":1080}
}

fpga_boards={
    "pynq-z2":{"dsp":220, "bram":280, "freq":0.2},
    "ultra96":{"dsp":360, "bram":432, "freq":0.2},
    "zcu104":{"dsp":1728/2, "bram":624/2, "freq":0.2},
    "zcu102":{"dsp":2520/2, "bram":1824/2, "freq":0.2},
    "v7":{"dsp":2020, "bram":2060, "freq": 0.1}
}


def ops(network):
    
    ops = 0
    for (l_name, l) in network.items():
        l_type = l['type']
        k = l['kernel']; nif = l['nif']; nof = l['nof']; nix = l['nix']
        stride = l['stride']**2 if 'stride' in l.keys() else 1
        if l_type == 'conv' or l_type == 'fc':
            print('name = {}, nif = {}, nof = {}, ops = {}'.format(l_name, nif, nof, ((k*k*nif*nof*nix*nix)/stride)*10**-9))
            ops += ((k*k*nif*nof*nix*nix)/stride)
    
    return ops * (10**(-9))