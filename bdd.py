import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (640,640)
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 0.0
        self.perspective = 0.0
        self.mosaic_prob = 0.5
        self.enable_mixup = True
        self.output_dir = "YOLOX_outputs"

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        base_path = "/mnt/disk2/soum/YoloX-HVAC/bdd_data/train_ready/"
        print("Getting ready")

        self.data_dir = base_path
        self.train_ann = f"train.json"
        self.val_ann = f"val.json"
        self.test_ann = f"val.json"
        self.num_classes = 10

        self.max_epoch = 500
        self.data_num_workers = 4
        self.eval_interval = 10

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
