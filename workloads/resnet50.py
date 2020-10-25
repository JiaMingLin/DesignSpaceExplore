def network(res, num_class = 101):
    nix = res["nix"]
    niy = res["niy"]
    network_template = {
        "conv1": {"nix": nix, "niy": niy, "nif": 3, "nof": 64, "stride":2, "kernel": 7, "type": "conv"}
        ############################# block 1 #############################
        "conv1_1": {"nix": nix/2, "niy": niy/2, "nif": 64, "nof": 64, "kernel": 1, "type": "pw_conv"}, #0
        "conv1_2": {"nix": nix/2, "niy": niy/2, "nif": 64, "nof": 64, "kernel": 3,  "type": "conv"},#1
        "conv1_3": {"nix": nix/2, "niy": niy/2, "nif": 64, "nof": 256, "kernel": 1, "type": "pw_conv"}, #2

        "conv2_1": {"nix": nix/2, "niy": niy/2, "nif": 256, "nof": 64, "kernel": 1, "type": "pw_conv"}, #0
        "conv2_2": {"nix": nix/2, "niy": niy/2, "nif": 64, "nof": 64, "kernel": 3,  "type": "conv"},#1
        "conv2_3": {"nix": nix/2, "niy": niy/2, "nif": 64, "nof": 256, "kernel": 1, "type": "pw_conv"}, #2

        "conv3_1": {"nix": nix/2, "niy": niy/2, "nif": 256, "nof": 64, "kernel": 1, "type": "pw_conv"}, #0
        "conv3_2": {"nix": nix/2, "niy": niy/2, "nif": 64, "nof": 64, "kernel": 3,  "stride": 2, "type": "conv"},#1
        "conv3_3": {"nix": nix/2, "niy": niy/2, "nif": 64, "nof": 256, "kernel": 1, "type": "pw_conv"}, #2

        ############################# block 2 #############################
        "conv4_1": {"nix": nix/4, "niy": niy/4, "nif": 256, "nof": 128, "kernel": 1, "type": "pw_conv"}, #0
        "conv4_2": {"nix": nix/4, "niy": niy/4, "nif": 128, "nof": 128, "kernel": 3,  "type": "conv"},#1
        "conv4_3": {"nix": nix/4, "niy": niy/4, "nif": 128, "nof": 512, "kernel": 1, "type": "pw_conv"}, #2

        "conv5_1": {"nix": nix/4, "niy": niy/4, "nif": 512, "nof": 128, "kernel": 1, "type": "pw_conv"}, #0
        "conv5_2": {"nix": nix/4, "niy": niy/4, "nif": 128, "nof": 128, "kernel": 3,  "type": "conv"},#1
        "conv5_3": {"nix": nix/4, "niy": niy/4, "nif": 128, "nof": 512, "kernel": 1, "type": "pw_conv"}, #2

        "conv6_1": {"nix": nix/4, "niy": niy/4, "nif": 512, "nof": 128, "kernel": 1, "type": "pw_conv"}, #0
        "conv6_2": {"nix": nix/4, "niy": niy/4, "nif": 128, "nof": 128, "kernel": 3,  "type": "conv"},#1
        "conv6_3": {"nix": nix/4, "niy": niy/4, "nif": 128, "nof": 512, "kernel": 1, "type": "pw_conv"}, #2

        "conv7_1": {"nix": nix/4, "niy": niy/4, "nif": 512, "nof": 128, "kernel": 1, "type": "pw_conv"}, #0
        "conv7_2": {"nix": nix/4, "niy": niy/4, "nif": 128, "nof": 128, "kernel": 3, "stride": 2, "type": "conv"},#1
        "conv7_3": {"nix": nix/4, "niy": niy/4, "nif": 128, "nof": 512, "kernel": 1, "type": "pw_conv"}, #2

        ############################# block 3 #############################
        "conv8_1": {"nix": nix/8, "niy": niy/8, "nif": 512, "nof": 256, "kernel": 1, "type": "pw_conv"}, #0
        "conv8_2": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 256, "kernel": 3,  "type": "conv"},#1
        "conv8_3": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 1024, "kernel": 1, "type": "pw_conv"}, #2

        "conv9_1": {"nix": nix/8, "niy": niy/8, "nif": 1024, "nof": 256, "kernel": 1, "type": "pw_conv"}, #0
        "conv9_2": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 256, "kernel": 3,  "type": "conv"},#1
        "conv9_3": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 1024, "kernel": 1, "type": "pw_conv"}, #2

        "conv10_1": {"nix": nix/8, "niy": niy/8, "nif": 1024, "nof": 256, "kernel": 1, "type": "pw_conv"}, #0
        "conv10_2": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 256, "kernel": 3,  "type": "conv"},#1
        "conv10_3": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 1024, "kernel": 1, "type": "pw_conv"}, #2

        "conv11_1": {"nix": nix/8, "niy": niy/8, "nif": 1024, "nof": 256, "kernel": 1, "type": "pw_conv"}, #0
        "conv11_2": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 256, "kernel": 3,  "type": "conv"},#1
        "conv11_3": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 1024, "kernel": 1, "type": "pw_conv"}, #2

        "conv12_1": {"nix": nix/8, "niy": niy/8, "nif": 1024, "nof": 256, "kernel": 1, "type": "pw_conv"}, #0
        "conv12_2": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 256, "kernel": 3,  "type": "conv"},#1
        "conv12_3": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 1024, "kernel": 1, "type": "pw_conv"}, #2

        "conv13_1": {"nix": nix/8, "niy": niy/8, "nif": 1024, "nof": 256, "kernel": 1, "type": "pw_conv"}, #0
        "conv13_2": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 256, "kernel": 3,  "stride": 2, "type": "conv"},#1
        "conv13_3": {"nix": nix/8, "niy": niy/8, "nif": 256, "nof": 1024, "kernel": 1, "type": "pw_conv"}, #2

        ############################# block 4 #############################
        "conv14_1": {"nix": nix/16, "niy": niy/16, "nif": 1024, "nof": 512, "kernel": 1, "type": "pw_conv"}, #0
        "conv14_2": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 512, "kernel": 3,  "type": "conv"},#1
        "conv14_3": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 2048, "kernel": 1, "type": "pw_conv"}, #2

        "conv15_1": {"nix": nix/16, "niy": niy/16, "nif": 1024, "nof": 512, "kernel": 1, "type": "pw_conv"}, #0
        "conv15_2": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 512, "kernel": 3,  "type": "conv"},#1
        "conv15_3": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 2048, "kernel": 1, "type": "pw_conv"}, #2

        "conv15_1": {"nix": nix/16, "niy": niy/16, "nif": 1024, "nof": 512, "kernel": 1, "type": "pw_conv"}, #0
        "conv15_2": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 512, "kernel": 3, "stride": 2, "type": "conv"},#1
        "conv15_3": {"nix": nix/16, "niy": niy/16, "nif": 512, "nof": 2048, "kernel": 1, "type": "pw_conv"}, #2

        # "fc1": {"nix": 1, "niy": 1, "nif": int(niy/32)*int(niy/32)*64, "nof": 512, "kernel": 1, "type":"fc"}, #13
        # "fc2": {"nix": 1, "niy": 1, "nif": 512, "nof": 4096, "kernel": 1, "type":"fc"}, #14
    }
    return network_template