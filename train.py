# import pandas as pd
# from sklearn.model_selection import train_test_split
# from transformers import BertModel
# from kobert_tokenizer import KoBERTTokenizer

# def preprocessing():
#     data = pd.read_csv(f'./data/data.csv', sep='|')
#     data_list = []
#     for q, label in zip(data['text'], data['score']):
#         data = []
#         data.append(q)
#         data.append(str(label))

#         data_list.append(data)

#     dataset_train, dataset_test = train_test_split(data_list, test_size=0.23, random_state=42)