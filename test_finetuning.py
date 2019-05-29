#%%
import argparse
import os
import time
import json
import autolab_core.utils as utils
from autolab_core import YamlConfig, Logger
from gqcnn import get_gqcnn_model, get_gqcnn_trainer, utils as gqcnn_utils

#%%
logger = Logger.get_logger('tools/finetune.py')

#%%
dataset_dir = '/home/ai/git/gqcnn/data/training/Dexnet-2.0_testTraining'
train_config = YamlConfig('cfg/finetune_dex-net_2.0_test.yaml')
gqcnn_params = train_config['gqcnn']

#%%
print(os.path.join(dataset_dir, 'config.json'))
config_filename = os.path.join(dataset_dir, 'config.json')
print(config_filename)
# print(os.getcwd())

#%%
open(config_filename, 'r')
config = json.load(open(config_filename, 'r'))

#%%
start_time = time.time()
gqcnn = get_gqcnn_model('tf')(gqcnn_params)
#%%
trainer = get_gqcnn_trainer('tf')(gqcnn, 'data/training/Dexnet-2.0_testFinetuning', 'image_wise', 'models/', train_config, 'GQCNN-2.0_finetuned_org')
# trainer = get_gqcnn_trainer('tf')(gqcnn, 'data/training/Dexnet-2.0_testTraining', 'image_wise', 'models/',train_config, 'GQCNN-2.0_Training_from_Scratch')

#%%
trainer.finetune('models/GQCNN-2.0_org')
logger.info('Total Fine-tuning Time: ' + str(utils.get_elapsed_time(time.time() - start_time)))