import os
from argparse import Namespace
from src.datamodule import DataModule
from src.model import ResNet
from src.trainer import Trainer


params = Namespace()
params.mode              = 'inference'
params.batch_size        = 100
params.resize            = (375, 375)
params.rand_noise_sigma  = 0.02
params.rand_noise_prob   = 0.1
params.data_root         = '/data/S/LinGroup/Users/sam/VRDL_HW1'
params.save_path         = os.path.join('/data/S/LinGroup/Users/sam/VRDL_HW1', 'checkpoint')
params.test_file         = ["testing_images"]


if __name__=='__main__':
    dataset = DataModule(params)
    ModelList = [
    #ResNet(params, ckpt="ep=041-acc=0.7967"),  #
    #ResNet(params, ckpt="ep=046-acc=0.7967"),  #
    ResNet(params, ckpt="ep=049-acc=0.8050"),  #
    #ResNet(params, ckpt="ep=060-acc=0.7933"),  #
    #ResNet(params, ckpt="ep=055-acc=0.7967"),  #
    #ResNet(params, ckpt="ep=060-acc=0.7967"),  #
    #ResNet(params, ckpt="ep=060-acc=0.7983"),
    ResNet(params, ckpt="ep=063-acc=0.8000"),
    ResNet(params, ckpt="ep=064-acc=0.8017"),
    #ResNet(params, ckpt="ep=066-acc=0.7983"),
    ResNet(params, ckpt="ep=071-acc=0.8050"),
    ResNet(params, ckpt="ep=076-acc=0.8033"),
    #ResNet(params, ckpt="ep=077-acc=0.7967"),
    ResNet(params, ckpt="ep=085-acc=0.8033"),
    #ResNet(params, ckpt="ep=089-acc=0.7983"),  #
    ResNet(params, ckpt="ep=098-acc=0.8000"),
    #ResNet(params, ckpt="ep=099-acc=0.7967"),  #
    #ResNet(params, ckpt="ep=080-acc=0.7967"),  #
    ResNet(params, ckpt="ep=080-acc=0.8000"),
    ResNet(params, ckpt="ep=081-acc=0.8033"),  #
    #ResNet(params, ckpt="ep=087-acc=0.7967"),  #
    #ResNet(params, ckpt="ep=100-acc=0.7917"),   ###
    ResNet(params, ckpt="ep=029-acc=0.8000"),
    ResNet(params, ckpt="ep=055-acc=0.8000"),
    ResNet(params, ckpt="ep=080-acc=0.8017"),
    ResNet(params, ckpt="ep=097-acc=0.8033"),
    ResNet(params, ckpt="ep=085-acc=0.8000")
    ]
    trainer = Trainer(params)
    trainer.ensemble(ModelList, dataset)
