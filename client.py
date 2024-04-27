import torch
import transformers
from torch import optim


class Client:
    def __init__(self, client_id, dataloader, opt, device):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.__model = None
        self.opt = opt

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model.to(self.device)

    def client_update(self, text2sql_tokenizer):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataloader.dataset)


class FedAvgClient(Client):
    def get_model_params(self):
        pass

    def client_update(self, text2sql_tokenizer):
        total_loss = 0

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.opt.learning_rate
        )

        print("[client", self.client_id, "] start local training")
        train_step = 0
        for epoch in range(1):
            print("[client", self.client_id, '] train lr', optimizer.state_dict()['param_groups'][0]['lr'])
            for batch in self.dataloader:
                train_step += 1

                batch_inputs = [data[0] for data in batch]
                batch_sqls = [data[1] for data in batch]
                batch_db_ids = [data[2] for data in batch]  # unused
                batch_tc_original = [data[3] for data in batch]  # unused

                # if epoch == 0:
                #     for batch_id in range(len(batch_inputs)):
                #         print(batch_inputs[batch_id])
                #         print(batch_sqls[batch_id])
                #         print("----------------------")

                tokenized_inputs = text2sql_tokenizer(
                    batch_inputs,
                    padding="max_length",
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )

                with text2sql_tokenizer.as_target_tokenizer():
                    tokenized_outputs = text2sql_tokenizer(
                        batch_sqls,
                        padding="max_length",
                        return_tensors='pt',
                        max_length=256,
                        truncation=True
                    )

                encoder_input_ids = tokenized_inputs["input_ids"]
                encoder_input_attention_mask = tokenized_inputs["attention_mask"]

                decoder_labels = tokenized_outputs["input_ids"]
                decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                decoder_attention_mask = tokenized_outputs["attention_mask"]

                encoder_input_ids = encoder_input_ids.to(self.device)
                encoder_input_attention_mask = encoder_input_attention_mask.to(self.device)
                decoder_labels = decoder_labels.to(self.device)
                decoder_attention_mask = decoder_attention_mask.to(self.device)

                model_outputs = self.model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_input_attention_mask,
                    labels=decoder_labels,
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict=True
                )
                loss = model_outputs["loss"]
                total_loss += loss.item()
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

        self.model.to("cpu")
        print("[client", self.client_id, "] local training finished, loss:", total_loss)
