import os
import pickle


class Util:

    @classmethod
    def save(cls, data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode='wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, mode='rb') as f:
            data = pickle.load(f)
        return data
