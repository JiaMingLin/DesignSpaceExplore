from .common import ops
def network(res, num_class = 101):
    nix = res["nix"]
    niy = res["niy"]
    network_template = {
        "conv1": {"nix": nix, "niy": niy, "nif": 3, "nof": 64, "stride":2, "kernel": 7, "type": "conv"},
        ############################# block 1 #############################
        "conv2_1": {"nix": nix/4, "niy": niy/4, "nif": 64, "nof": 64, "kernel": 3,  "type": "conv"},#1
        "conv2_2": {"nix": nix/4, "niy": niy/4, "nif": 64, "nof": 64, "kernel": 3,  "type": "conv"},#1
        
        "conv3_1": {"nix": nix/4, "niy": niy/4, "nif": 64, "nof": 64, "kernel": 3,  "type": "conv"},#1
        "conv3_2": {"nix": nix/4, "niy": niy/4, "nif": 64, "nof": 64, "kernel": 3, "type": "conv"},#1
        "conv3_3": {"nix": nix/4, "niy": niy/4, "nif": 64, "nof": 128, "kernel": 3, "stride": 2, "type": "conv"},#1
        
        ############################# block 2 #############################
        "conv4_1": {"nix": nix/16, "niy": niy/16, "nif": 128, "nof": 128, "kernel": 3,  "type": "conv"},#1
        "scale_conv_1": {"nix": nix/4, "niy": niy/4, "nif": 64, "nof": 128, "kernel": 1, "stride": 2, "type": "conv"},#1

        "conv5_1": {"nix": nix/16, "niy": niy/16, "nif": 128, "nof": 128, "kernel": 3,  "type": "conv"},#1
        "conv5_2": {"nix": nix/16, "niy": niy/16, "nif": 128, "nof": 128, "kernel": 3,  "type": "conv"},#1
        "conv5_3": {"nix": nix/16, "niy": niy/16, "nif": 128, "nof": 256, "kernel": 3,  "stride": 2,  "type": "conv"},#1
        
        ############################# block 3 #############################
        "conv6_1": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 256, "kernel": 3,  "type": "conv"},#1
        "conv6_2": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 256, "kernel": 3,  "type": "conv"},#1
        
        "conv7_1": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 256, "kernel": 3,  "type": "conv"},#1
        "conv7_2": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 512, "kernel": 3,  "stride": 2, "type": "conv"},#1

        ############################# block 4 #############################
        "conv8_1": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 512, "kernel": 3, "type": "conv"},#1
        "conv8_2": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 512, "kernel": 3, "type": "conv"},#1
        
        "conv9_1": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 512, "kernel": 3, "type": "conv"},#1
        "conv9_2": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 512, "kernel": 3, "stride": 2, "type": "conv"},#1

        "fc1": {"nix": 1, "niy": 1, "nif": int(niy/32)*int(niy/32)*512, "nof": num_class, "kernel": 1, "type":"fc"}, #13
    }
    operations = ops(network_template)
    print("ResNet-18 OPs = ", operations)
    return network_template