import copy
import multiprocessing
from collections import OrderedDict

import torch
from tokenizers import AddedToken
from transformers import T5TokenizerFast


def train_client(client, tokenizer):
    client.client_update(tokenizer)


class CenterServer:
    def __init__(self, model, opt, device="cuda:0"):
        self.aggregation_weights = None
        self.total_data_size = None
        self.model = model
        self.device = device
        self.clients = []
        self.opt = opt
        text2sql_tokenizer = T5TokenizerFast.from_pretrained(
            opt.model_name_or_path,
            add_prefix_space=True
        )

        if isinstance(text2sql_tokenizer, T5TokenizerFast):
            text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
        self.text2sql_tokenizer = text2sql_tokenizer

    def add_client(self, client):
        self.clients.append(client)
        self.total_data_size = sum([len(client) for client in self.clients])
        self.aggregation_weights = [
            len(client) / self.total_data_size for client in self.clients
        ]

    def aggregation(self):
        raise NotImplementedError

    def start_local_train(self):
        print("[Server] Starting local training")
        processes = []
        for client in self.clients:
            p = multiprocessing.Process(target=train_client, args=(client, self.text2sql_tokenizer))
            p.start()
            processes.append(p)

            # 等待所有进程完成
        for p in processes:
            p.join()
        print("[Server] All local training finished")

    def send_model(self):
        print("[Server] Sending model to local")
        for client in self.clients:
            client.model = copy.deepcopy(self.model)

    def validation(self):
        pass

    def save(self, epoch):
        self.model.save_pretrained(
            save_directory=self.opt.save_path + "/fed-lr{}-epoch{}-client_num{}".format(self.opt.learning_rate, epoch, self.opt.client_num))
        self.text2sql_tokenizer.save_pretrained(
            save_directory=self.opt.save_path + "/fed-lr{}-epoch{}-client_num{}".format(self.opt.learning_rate, epoch, self.opt.client_num))


class FedAvgCenterServer(CenterServer):
    def __init__(self, model, dataloader, device="cuda:0"):
        super().__init__(model, dataloader, device)

    def aggregation(self):
        print("[Server] aggregation start")
        update_state = OrderedDict()
        aggregation_device = self.model.device

        for k, client in enumerate(self.clients):
            local_state = {key: value.to(aggregation_device) for key, value in client.model.state_dict().items()}

            for key in self.model.state_dict().keys():
                if k == 0:
                    update_state[key] = local_state[key] * self.aggregation_weights[k]
                else:
                    update_state[key] += local_state[key] * self.aggregation_weights[k]

        self.model.load_state_dict(update_state)
        print("[Server] aggregation finished")
