import tensorflow as tf
import tensorflow_text as text
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU') # Set this to use CPU and avoid error msg about cuda dlls

__version__ = "0.1.0"
BASE_DIR = Path(__file__).resolve(strict=True).parent
class BertModel():
    def __init__(self):
        self.model = self.load()

    def load(self):
        path = '.'
        reloaded_model = tf.saved_model.load(f"{BASE_DIR}/{path}")
        print("Model Loaded correctly")
        return reloaded_model
        

    def print_bert_results(self, bert_result):
        print(bert_result)
        print(tf.nn.softmax(bert_result))
        bert_result_class = tf.argmax(bert_result, axis=1)[0]

        if bert_result_class == 1:
            return 'acceptable'
        else:
            return 'unacceptable'

    def predict(self, word):
        tfword = tf.constant([word])

        result = self.model(tfword)
        return self.print_bert_results(result)