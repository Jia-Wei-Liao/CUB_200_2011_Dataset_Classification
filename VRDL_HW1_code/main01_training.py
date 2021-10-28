import os
from argparse import Namespace
from src.datamodule import DataModule
from src.model import ResNet
from src.trainer import Trainer


params = Namespace()
params.mode              = 'training'
params.model             = 'resnet50'
params.fine_tune         = False
params.max_epochs        = 100
params.batch_size        = 20
params.train_num         = 2400
params.valid_num         = 600
params.optimizer         = 'AdamW'
params.lr                = 1e-4
params.lr_scheduler      = 'step'
params.lr_decay_period   = 3
params.lr_decay_factor   = 0.8
params.weight_decay      = 1e-4
params.resize            = (375, 375)
params.rand_noise_sigma  = 0.02
params.rand_noise_prob   = 0.1
params.baseline          = 0.79
params.train_file        = ["training_images", "training_images_aug1", "training_images_aug2", "training_images_aug3"]
params.valid_file        = ["training_images"]
params.data_root         = '/data/S/LinGroup/Users/sam/VRDL_HW1'
params.save_path         = os.path.join('/data/S/LinGroup/Users/sam/VRDL_HW1', 'checkpoint')


if __name__=='__main__':
    dataset = DataModule(params)
    model = ResNet(params)
    trainer = Trainer(params)
    trainer.fit(model, dataset)