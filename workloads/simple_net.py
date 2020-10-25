def network(res, num_class = 101):
    nix = res["nix"]
    niy = res["niy"]
    network_template = {
        "conv1_1": {"nix": nix, "niy": niy, "nif": 3, "nof": 32, "kernel": 3, "type": "conv"}, #0
        "conv1_2": {"nix": nix/2, "niy": niy/2, "nif": 32, "nof": 64, "kernel": 3,  "type": "conv"},#1
        "conv2_1": {"nix": nix/4, "niy": niy/4, "nif": 64, "nof": 64, "kernel": 3, "type": "conv"}, #2
        "conv2_2": {"nix": nix/8, "niy": niy/8, "nif": 64, "nof": 64, "kernel": 3, "type": "conv"}, #3
        "conv3_1": {"nix": nix/16, "niy": niy/16, "nif": 64, "nof": 64, "kernel": 3, "type": "conv"}, #4
        # "fc1": {"nix": 1, "niy": 1, "nif": int(niy/32)*int(niy/32)*64, "nof": 512, "kernel": 1, "type":"fc"}, #13
        # "fc2": {"nix": 1, "niy": 1, "nif": 512, "nof": 4096, "kernel": 1, "type":"fc"}, #14
    }
    return network_template