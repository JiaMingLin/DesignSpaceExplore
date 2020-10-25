def network(res, num_class = 1000):
    nix = res["nix"]
    niy = res["niy"]
    network_template = {
        "conv1_1": {"nix": nix, "niy": niy, "nif": 3, "nof": 64, "kernel": 3, "type": "conv"}, #0
        "conv1_2": {"nix": nix, "niy": niy, "nif": 64, "nof": 64, "kernel": 3,  "type": "conv"},#1
        "conv2_1": {"nix": nix/2, "niy": niy/2, "nif": 64, "nof": 128, "kernel": 3, "type": "conv"}, #2
        "conv2_2": {"nix": nix/2, "niy": niy/2, "nif": 128, "nof": 128, "kernel": 3, "type": "conv"}, #3
        "conv3_1": {"nix": nix/4, "niy": niy/4, "nif": 128, "nof": 256, "kernel": 3, "type": "conv"}, #4
        "conv3_2": {"nix": nix/4, "niy": niy/4, "nif": 256, "nof": 256, "kernel": 3, "type": "conv"}, #5
        "conv3_3": {"nix": nix/4, "niy": niy/4, "nif": 256, "nof": 256, "kernel": 3, "type": "conv"}, #6
        "conv4_1": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 512, "kernel": 3, "type": "conv"}, #7
        "conv4_2": {"nix": nix/8, "niy": niy/8, "nif": 512, "nof": 512, "kernel": 3, "type": "conv"}, #8
        "conv4_3": {"nix": nix/8, "niy": niy/8, "nif": 512, "nof": 512, "kernel": 3, "type": "conv"}, #9
        "conv5_1": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 512, "kernel": 3, "type": "conv"}, #10
        "conv5_2": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 512, "kernel": 3, "type": "conv"}, #11
        "conv5_3": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 512, "kernel": 3, "type": "conv"}, #12
        #"fc6": {"nix": nix/32, "niy": niy/32, "nif": 512, "nof": 4096, "kernel": 1, "type":"fc"}, #13
        #"fc7": {"nix": 1, "niy": 1, "nif": 4096, "nof": 4096, "kernel": 1, "type":"fc"}, #14
        #"fc8": {"nix": 1, "niy": 1, "nif": 4096, "nof": num_class, "kernel": 1, "type":"fc"}, #15
    }
    return network_template
