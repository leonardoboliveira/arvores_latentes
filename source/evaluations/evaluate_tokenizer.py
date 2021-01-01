from data_handling.create_bert_ts import create_tokenizer

tokenizer = create_tokenizer()

tokens = tokenizer.tokenize("Money Market Deposits - a 6.21 %")

print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)