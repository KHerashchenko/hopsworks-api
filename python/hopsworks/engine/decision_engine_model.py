import random
from typing import List, Dict

import tensorflow as tf
import joblib

class ItemCatalogEmbedding(tf.keras.Model):
    """
    Candidate embedding tower of the Retrieval model
    """

    def __init__(
        self,
        configs_dict: dict,
        pk_index_list: List[str],
        categories_lists: Dict[str, List[str]],
    ):
        super().__init__()

        self._configs_dict = configs_dict
        item_space_dim = self._configs_dict["model_configuration"]["retrieval_model"][
            "item_space_dim"
        ]

        self.pk_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(vocabulary=pk_index_list, mask_token=None),
                tf.keras.layers.Embedding(
                    # We add an additional embedding to account for unknown tokens.
                    len(pk_index_list) + 1,
                    item_space_dim,
                ),
            ]
        )

        self.categories_tokenizers = {}
        self.categories_lens = {}
        for feat, lst in categories_lists.items():
            self.categories_tokenizers[feat] = tf.keras.layers.StringLookup(
                vocabulary=lst, mask_token=None
            )
            self.categories_lens[feat] = len(lst)

        vocab_size = 1000
        # self.texts_embeddings = {}
        self.normalized_feats = {}
        for feat, val in self._configs_dict["product_list"]["schema"].items():
            if "transformation" not in val.keys():
                continue
            # if val["transformation"] == "text":
            #     self.texts_embeddings[feat] = tf.keras.Sequential(
            #         [
            #             tf.keras.layers.TextVectorization(
            #                 max_tokens=vocab_size,
            #             ),
            #             tf.keras.layers.Embedding(
            #                 vocab_size + 1, item_space_dim, mask_zero=True
            #             ),
            #             tf.keras.layers.GlobalAveragePooling1D(),
            #         ]
            #     )
            if val["transformation"] in [
                "numeric",
                "timestamp",
            ]:  # TODO change feature engineering for timestamps cause this is fucked
                self.normalized_feats[feat] = tf.keras.layers.Normalization(axis=None)

        self.fnn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(item_space_dim, activation="relu"),
                tf.keras.layers.Dense(item_space_dim),
            ]
        )

    def call(self, inputs):
        # Explicitly name input tensors
        pk_inputs = inputs[self._configs_dict["product_list"]["primary_key"]]
        category_inputs = {feat: inputs[feat] for feat in self.categories_tokenizers}
        # text_inputs = {feat: inputs[feat] for feat in self.texts_embeddings} # TODO couldnt solve errors with Pooling layer
        numeric_inputs = {feat: inputs[feat] for feat in self.normalized_feats}

        layers = [self.pk_embedding(pk_inputs)]

        for feat, val in self._configs_dict["product_list"]["schema"].items():
            if "transformation" not in val.keys():
                continue
            if val["transformation"] == "category":
                layers.append(
                    tf.one_hot(
                        self.categories_tokenizers[feat](category_inputs[feat]),
                        self.categories_lens[feat],
                    )
                )
            # elif val["transformation"] == "text":
            #     layers.append(self.texts_embeddings[feat](tf.expand_dims(text_inputs[feat], 0)))
            elif val["transformation"] in ["numeric", "timestamp"]:
                tensor = tf.reshape(
                    self.normalized_feats[feat](numeric_inputs[feat]), (-1, 1)
                )
                layers.append(tensor)

        print("Layers are:", layers)
        # concatenated_inputs = tf.concat(adjusted_layers, axis=-1) # TODO cant fix dimensions
        outputs = self.fnn(layers[0])
        return outputs

class QueryModel(tf.keras.Model):
    def __init__(self, vocabulary, item_space_dim):
        super().__init__()
        self.query_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=vocabulary, mask_token=None),
            tf.keras.layers.Embedding(
                len(vocabulary) + 1, item_space_dim
            ),
            tf.keras.layers.GRU(item_space_dim)
        ])
        
    def call(self, inputs):
        context_input = tf.reshape(inputs["context_item_ids"], [-1, 10])
        return self.query_model(context_input)
    
    
class RandomPredictor:
    def predict(self, inputs):
        n = len(inputs[0])
        return [[random.random()] for _ in range(n)]
    
    def save(self, filepath):
        # Save the model using joblib
        joblib.dump(self, filepath)