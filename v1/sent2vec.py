import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tensorflow_hub as hub
import pickle
from data import DATASET_CACHE

EMBED_CACHE = "embed_cache.pkl"

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module("./elmo", name='elmo_module_003')
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

if __name__ == '__main__':
    with open(DATASET_CACHE, "rb") as dataset_file:
        all_comments, labels = pickle.load(dataset_file)

    embed_fn = embed_useT('../sentence_wise_email/module/module_useT')
    sent_embeddings = embed_fn(all_comments)
    print(sent_embeddings.shape)

    with open(EMBED_CACHE, "wb") as embed_file:
        pickle.dump([sent_embeddings, labels], embed_file)
