import copy
import multiprocessing
import os
import time

import torch
import argparse

from tokenizers import AddedToken
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.trainer_utils import set_seed
from utils.load_dataset import Text2SQLDataset
from server import FedAvgCenterServer
from client import FedAvgClient


def simple_collate(batch):
    return batch

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")

    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size.')
    parser.add_argument('--gradient_descent_step', type=int, default=4,
                        help='perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--device', type=str, default="2",
                        help='the id of used GPU device.')
    parser.add_argument('--learning_rate', type=float, default=3e-5,
                        help='learning rate.')
    parser.add_argument('--epochs', type=int, default=128,
                        help='training epochs.')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed.')
    parser.add_argument('--save_path', type=str, default="models/text2sql",
                        help='save path of best fine-tuned text2sql model.')
    parser.add_argument('--tensorboard_save_path', type=str, default="tensorboard_log/text2sql",
                        help='save path of tensorboard log.')
    parser.add_argument('--model_name_or_path', type=str, default="t5-base",
                        help=
                        '''
                        pre-trained model name. 
                        options: 
                            t5-base, https://huggingface.co/t5-base;
                            t5-large, https://huggingface.co/t5-large;
                            t5-3b, https://huggingface.co/t5-3b;
                        ''')
    parser.add_argument('--use_adafactor', action='store_true',
                        help='whether to use adafactor optimizer.')
    parser.add_argument('--mode', type=str, default="train",
                        help='trian, eval or test.')
    parser.add_argument('--train_filepath', type=str, default="data/preprocessed_data/resdsql_train_spider.json",
                        help='file path of test2sql training set.')
    parser.add_argument('--dev_filepath', type=str, default="data/preprocessed_data/resdsql_dev.json",
                        help='file path of test2sql dev set.')
    parser.add_argument('--original_dev_filepath', type=str, default="data/spider/dev.json",
                        help='file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type=str, default="database",
                        help='file path of database.')
    parser.add_argument('--tables_for_natsql', type=str, default="NatSQL/NatSQLv1_6/tables_for_natsql.json",
                        help='file path of tables_for_natsql.json.')
    parser.add_argument('--num_beams', type=int, default=8,
                        help='beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type=int, default=8,
                        help='the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--target_type", type=str, default="sql",
                        help="sql or natsql.")
    parser.add_argument("--output", type=str, default="predicted_sql.txt",
                        help="save file of the predicted sqls.")
    parser.add_argument("--client_num", type=int, default="5",
                        help="the number of clients")

    opt = parser.parse_args()

    return opt


def _train(opt):
    client_num = opt.client_num

    set_seed(opt.seed)
    print(opt)

    if opt.tensorboard_save_path is not None:
        writer = SummaryWriter(opt.tensorboard_save_path)
    else:
        writer = None

    text2sql_tokenizer = T5TokenizerFast.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space=True
    )

    if isinstance(text2sql_tokenizer, T5TokenizerFast):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    train_dataset = Text2SQLDataset(
        dir_=opt.train_filepath,
        mode="train"
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else T5ForConditionalGeneration

    print("initializing text2sql model.")
    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path)
    model.resize_token_embeddings(len(text2sql_tokenizer))
    if torch.cuda.is_available():
        model = model.cuda()
    print("finished.")

    # init server

    server = FedAvgCenterServer(model, opt)

    for client_id in range(client_num):
        local_dataset = copy.deepcopy(train_dataset).to_fed_dataset(client_id, client_num)
        train_dataloder = DataLoader(
            local_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            collate_fn=simple_collate,
            drop_last=True
        )
        device = "cuda:" + str(client_id)
        client = FedAvgClient(client_id, train_dataloder, opt, device)
        server.add_client(client)

    for global_epoch in range(opt.epochs):
        start_time = time.time()

        server.send_model()
        server.start_local_train()
        server.aggregation()

        end_time = time.time()
        epoch_duration = end_time - start_time
        print("global epoch", global_epoch, "finish in", epoch_duration, "s")
        print("-------------------")
        if global_epoch % 20 == 19:
            server.save(global_epoch)




if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    opt = parse_option()
    if opt.mode in ["train"]:
        _train(opt)
