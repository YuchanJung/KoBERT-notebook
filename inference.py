import os.path as p
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gluonnlp as nlp

from torch.utils.data import Dataset, DataLoader
from config import config
from model import BERTClassifier
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
from data import BERTDataset

device = torch.device("cuda:0")
# device = torch.device("cpu")

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token="[PAD]")
tok = tokenizer.tokenize

bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
model.load_state_dict(torch.load(f'./model/model_train85_test_76.pt', map_location=device))

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, config.max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=config.batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("IMMORAL_NONE")
            elif np.argmax(logits) == 1:
                test_eval.append("IMMORAL")
            elif np.argmax(logits) == 2:
                test_eval.append("IMMORAL_MAX")

        return test_eval[0]
    

while True:
    sentence = input("Input: ")
    if sentence == '0': break
    output = predict(sentence)
    print(f">> 입력하신 내용에서 {output}이/가 느껴집니다.")
