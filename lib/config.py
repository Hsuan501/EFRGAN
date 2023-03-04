'''
Default config for EFR Gan
'''

from yacs.config import CfgNode as CN
import argparse
import os

cfg = CN()

abs_efrGan_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cfg.emotion_type = 'happy'
cfg.gan_dir = abs_efrGan_dir
cfg.device = 'cuda'
cfg.device_id = '0'

cfg.pretrained_modelpath = os.path.join(cfg.gan_dir, 'model_checkpoint', 'G_model.tar')
cfg.output_dir = os.path.join(cfg.gan_dir, 'model_checkpoint', f'Gan_model_{cfg.emotion_type}')
cfg.rasterizer_type = 'pytorch3d'
# num_classes = num_class +1 (disgust is 2)
cfg.num_classes = 3

# ---------------------------------------------------------------------------- #
# Options for Generator
# ---------------------------------------------------------------------------- #
cfg.G = CN()
# generator
cfg.G.pretrained_modelpath = os.path.join(cfg.gan_dir, 'model_checkpoint', 'G_model.tar')
cfg.G.fc_embedsize = 256
cfg.G.fc_hiddensize = 128
cfg.G.exp_outputsize = 50
cfg.G.pose_outputsize = 3
cfg.G.img_shape = (3,224,224)
cfg.G.lr = 5e-4

# flame model
cfg.G.flame_model_path = os.path.join(cfg.gan_dir, 'model_checkpoint', 'generic_model.pkl') 
cfg.G.flame_lmk_embedding_path = os.path.join(cfg.gan_dir, 'model_checkpoint', 'landmark_embedding.npy') 
cfg.G.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.G.n_shape = 100
cfg.G.n_tex = 50
cfg.G.n_exp = 50
cfg.G.n_cam = 3
cfg.G.n_pose = 6
cfg.G.n_light = 27
cfg.G.use_tex = False
cfg.G.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning
# face recognition model
cfg.G.fr_model_path = os.path.join(cfg.gan_dir, 'model_checkpoint', 'resnet50_ft_weight.pkl')


# ---------------------------------------------------------------------------- #
# Options for Discriminator
# ---------------------------------------------------------------------------- #
cfg.D = CN()
cfg.D.channels = 1
cfg.D.img_shape = (1,224,224)
cfg.D.lr = 1e-6
cfg.D.label_path = f'/home/taros/Documents/Flame/new_ERFGAN/dataset/{cfg.emotion_type}_label.txt'
# ---------------------------------------------------------------------------- #
# Options for train
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.max_epochs = 1

cfg.train.b1 = 0.5
cfg.train.b2 = 0.999
cfg.train.batch_size = 16
# "number of training steps for discriminator per iter"
cfg.train.n_critic = 5
cfg.train.lambda_gp = 10
cfg.train.vis_dir = 'train_images'
cfg.train.vis_steps = 10
cfg.train.log_dir = 'logs'
cfg.train.write_summary = True
cfg.train.checkpoint_steps = 100
cfg.train.resume = True

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.D_sample_dir = os.path.join(cfg.gan_dir, 'dataset/emotion3Dface/happy')
cfg.dataset.G_sample_dir = os.path.join(cfg.gan_dir, 'dataset/emotion3Dface/neutral')

# cfg.dataset.training_data = ['ethnicity']
cfg.dataset.eval_data = ['aflw2000']
cfg.dataset.test_data = ['']
cfg.dataset.batch_size = 2
cfg.dataset.K = 4
cfg.dataset.isSingle = False
cfg.dataset.num_workers = 2
cfg.dataset.image_size = 224
cfg.dataset.scale_min = 1.4
cfg.dataset.scale_max = 1.8
cfg.dataset.trans_scale = 0.

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--mode', type=str, default = 'train', help='deca mode')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    cfg.cfg_file = None
    cfg.mode = args.mode
    # import ipdb; ipdb.set_trace()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
