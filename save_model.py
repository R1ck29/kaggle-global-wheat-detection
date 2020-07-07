import os

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger

from src.lightning_classes.lightning_wheat import LitWheat
from src.utils.get_dataset import get_test_dataset
from src.utils.utils import set_seed, format_prediction_string, collate_fn


def save_model(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model

    Args:
        cfg: hydra config

    """
    # set_seed(cfg.training.seed)
    # model = LitWheat(hparams=cfg, cfg)

    ckpt_path = '/home/rick/Dropbox/python_projects/data_science/Kaggle/global_wheat_detection/wheat/outputs/2020-06-28/14-34-34/saved_models/_ckpt_epoch_9.ckpt'
    model = LitWheat.load_from_checkpoint(checkpoint_path=ckpt_path, cfg=cfg)

    # save as a simple torch model
    # model_name = os.getcwd().split('\\')[-1] + '.pt'
    model_name = '/home/rick/Dropbox/python_projects/data_science/Kaggle/global_wheat_detection/wheat/outputs/2020-06-28/14-34-34/saved_models/model006.pt'
    print(model_name)
    torch.save(model.model.state_dict(), model_name)
    

@hydra.main(config_path='conf/config.yaml')
def main(cfg: DictConfig) -> None:
    # print(cfg.pretty())
    save_model(cfg)


if __name__ == '__main__':
    main()