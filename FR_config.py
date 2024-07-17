import json

""" configuration json """

class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)

config = Config({

        # 1.win5
        "db_name":                      "win5",
        "db_path":                      "/home/usr2/Dataset/win5/Distorted",
        "Ref_path":                     "/home/usr2/Dataset/win5/reference/",
        "text_path":                    "/home/usr2/Dataset/win5/1.txt",    

        # 2.NBU
        # "db_name":                      "NBU",
        # "db_path":                      "/home/usr2/Dataset/NBU/NBU/dis_img/",
        # "Ref_path":                     "/home/usr2/Dataset/NBU/NBU/ref_img/",
        # "text_path":                    "/home/usr2/Dataset/NBU/NBU/1.txt",

        # 3.SHU
        # "db_name":                      "SHU",
        # "db_path":                      "/home/usr2/Dataset/SHU/dis/",
        # "Ref_path":                     "/home/usr2/Dataset/SHU/ref/",
        # "text_path":                    "/home/usr2/Dataset/SHU/1.txt",

        "svPath":                       "./result",
        "mos_sv_path":                  "./data",
        "model_path":                  "./model/resnet50.pth",

        "batch_size":                   1,                               # batch size
        "n_epoch":                      300,                             # epoch
        "val_freq":                     1,                               # test freq
        "crop_size":                    224,                             # patch size

        "aug_num":                      1,                               # patch num
        "if_avg":                       True,                            # average mark:True\False
        "avg_num":                      4,                               # average

        "train_rate":                   0.8,
        "normal_test":                  False,                           # normal test or five_point_crop, True\False
        "if_resize":                    True,                            # True\False

        "learning_rate":                5e-6,                            #
        "weight_decay":                 1e-5,                            # 1e-5, 0
        "T_max":                        50,                              # 50, 3e4
        "eta_min":                      0,
        "num_avg_val":                  5,
        "num_workers":                  0,                               # num_workers

        "input_size":                   (512, 512),                      # (512, 512),(434, 625)
        "new_size":                     (224, 224),
        "patch_size":                   16,                              # patch size
        "img_size":                     224,                             #

        "embed_dim":                    768,                             # 768, 192, 384, 1024
        "dim_mlp":                      768,                             # 768, ……
        "num_heads":                    [4, 4],                          #
        "window_size":                  2,                               # window_size
        "depths":                       [2, 2],                          # encoder
        "num_outputs":                  1,
        "num_tab":                      2,                               # attention block
        "scale":                        0.13,                            # 0.13

        # optimization & training parameters
        'lr_rate':                      1e-4,
        'momentum':                     0.9,

        # load & save checkpoint
        "model_name": "model",
        "output_path": "./output",
        "snap_path": "./output/models/",                                # directory for saving checkpoint
        "log_path": "./output/log/",
        "log_file": ".txt",
        "tensorboard_path": "./output/tensorboard/"
    })
