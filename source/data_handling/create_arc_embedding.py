from training.train_mentions import get_model
from data_handling.create_mention_embedding_ts import get_embeddings_for_file, to_example, MAX_NUM_MENTIONS
import numpy as np

_, encoder = get_model()
encoder.load_weights(r'D:\GDrive\Puc\Projeto Final\Datasets\arc\encoder.h5')
file_name = r"D:\ProjetoFinal\data\debug\train.conll.test.0"
for doc_id, embeddings, labels, mask, doc, cumsum in get_embeddings_for_file(file_name, ret_doc=True):
    embeddinds = []
    spans = []

    counter = 1
    for example, features in to_example(embeddings, labels, mask, ret_dict=True):
        input_embeddings = np.array(features["input_embeddings"].float_list.value).reshape((MAX_NUM_MENTIONS, -1))
        embeddinds.append(input_embeddings)
        length = sum(features["input_mask"].int64_list.value)
        spans.append((counter, length))
        counter += 1

    enc = encoder.predict(np.array(embeddinds))
    print(enc)
