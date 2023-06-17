import gensim
from gensim.models import word2vec

model_path = "embedding/word2vec_300.w2v"

# Load the pre-trained w2v file
word2vec_model = word2vec.Word2Vec.load(model_path)
