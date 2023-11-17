# Copyright 2019 Rui Qiao. All Rights Reserved.
#
# DeepNovoV2 is publicly available for non-commercial uses.
# ==============================================================================
import time
import torch
import logging
import logging.config
import deepnovo_config
import os
from train_func1 import train, build_model, validation, perplexity
# from train_func2 import train, build_model, validation, perplexity
# from train_func import train, build_model, validation, perplexity
from data_reader import DeepNovoDenovoDataset, collate_func, DeepNovoTrainDataset
from model_gcn import InferenceModelWrapper
from denovo1 import IonCNNDenovo
# from model import InferenceModelWrapper
# from denovo import IonCNNDenovo
# from model_gcn2 import InferenceModelWrapper
# from denovo2 import IonCNNDenovo
from writer import DenovoWriter
from init_args import init_args
import deepnovo_worker_test
# from deepnovo_dia_script_select import find_score_cutoff
import datetime
from TagWriter import TagWrite
from SpectralGraph import Tag
logger = logging.getLogger(__name__)

def engine_1(args):
    # train + search denovo + test
    start = time.time()
    logger.info(f"training mode")
    torch.cuda.empty_cache()
    train(args=args)
    logger.info(f'using time:{time.time() - start}')
    engine_2(args)


def engine_2(args):
    # search denovo + test
    """
    search denovo
    """
    torch.cuda.empty_cache()
    start = time.time()
    logger.info("denovo mode")
    data_reader = DeepNovoDenovoDataset(feature_filename=args.denovo_input_feature_file,
                                        spectrum_filename=args.denovo_input_spectrum_file,
                                        args=args)
    # denovo_worker = IonCNNDenovo(args=args)
    # forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
    # model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, init_net)
    forward_deepnovo, backward_deepnovo, init_net = build_model(args=args, training=False)
    model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, init_net)
    # writer = DenovoWriter(args=args)
    # denovo_worker.search_denovo(model_wrapper, data_reader, writer)
    torch.cuda.empty_cache()

    writer = TagWrite(args, logger)
    tag = Tag(args=args)
    tag.search(model_wrapper, data_reader, writer)

    logger.info(f'using time:{time.time() - start}')



def engine_3(args):
    # test
    logger.info("test mode")
    worker_test = deepnovo_worker_test.WorkerTest(args=args)
    worker_test.test_accuracy()

    # show 95 accuracy score threshold
    accuracy_cutoff = 0.95
    accuracy_file = args.accuracy_file
    # score_cutoff = find_score_cutoff(accuracy_file, accuracy_cutoff)


def init_log(log_file_name):
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)

if __name__ == '__main__':
    nine_species=[
        "bacillus",
        # "clambacteria",
        # "honeybee",
        # "human",
        # "mmazei",
        # "mouse",
        # "ricebean",
        # "tomato",
        # "yeast",
    ]
    param_path = [
        "./param/" \
        "cross.9high_80k.exclude_4_%s_PeakNetwork_NH3H2O_InternalIons_Edge_Charge2[2adj].cfg"%p for p in nine_species
    ]


    log_path = "./log/"
    if isinstance(param_path,list):
        print(param_path)
        for _param_path in param_path:
            dir, param_file = os.path.split(_param_path)
            # log_file_name = "top5_" + param_file[-4] + ".log"
            now = datetime.datetime.now().strftime("%Y%m%d%H%M")
            args = init_args(_param_path)
            # log_file_name = "./log/" + now + "(" + str(args.engine_model) + ").log"
            log_file_name = log_path + param_file + now + "(" + str(args.engine_model) + ").log"
            init_log(log_file_name=log_file_name)
            if os.path.exists(args.train_dir):
                pass
            else:
                os.makedirs(args.train_dir)
            if args.engine_model == 1:
                # print("engine model 1")
                engine_1(args=args)
            elif args.engine_model == 2:
                engine_2(args=args)
            elif args.engine_model == 3:
                engine_3(args=args)