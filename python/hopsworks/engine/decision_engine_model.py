from typing import List, Dict

import tensorflow as tf
import tensorflow_recommenders as tfrs

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


class QueryModelModule(tf.Module):
    def __init__(self, query_model):
        self.query_model = query_model

    @tf.function()
    def compute_emb(self, instances):
        # Compute the query embeddings
        query_emb = self.query_model(instances["context_item_ids"])
        # Ensure the output is a dictionary of tensors
        return {
            "query_emb": query_emb,
        }


class RankingModel(tf.keras.Model):
    """
    Session embedding model used in the Ranking model.
    """

    def __init__(
        self,
        configs_dict: dict,
        categories_lists: Dict[str, List[str]],
    ):
        super().__init__()

        # self.latitude = tf.keras.layers.Normalization(axis=None)
        # self.longitude = tf.keras.layers.Normalization(axis=None)

        # language_codes = [
        #     "en",
        #     "es",
        #     "fr",
        #     "de",
        #     "it",
        #     "pt",
        #     "nl",
        #     "sv",
        # ]  # TODO provide full list
        # self.language = tf.keras.layers.StringLookup(
        #     vocabulary=language_codes, mask_token=None
        # )
        # self.language_len = len(language_codes)

        self._configs_dict = configs_dict

        self.categories_tokenizers = {}
        self.categories_lens = {}
        for feat, lst in categories_lists.items():
            self.categories_tokenizers[feat] = tf.keras.layers.StringLookup(
                vocabulary=lst, mask_token=None
            )
            self.categories_lens[feat] = len(lst)

        self.normalized_feats = {}
        for feat, val in self._configs_dict["product_list"]["schema"].items():
            if "transformation" not in val.keys():
                continue
            if val["transformation"] in [
                "numeric",
                "timestamp",
            ]:  # TODO change feature engineering for timestamps cause this is fucked
                self.normalized_feats[feat] = tf.keras.layers.Normalization(axis=None)

        self.ratings = tf.keras.Sequential(
            [
                # Learn multiple dense layers.
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )

    def compute_candidate_embedding(self, inputs):
        # category_inputs = {feat: inputs[feat] for feat in self.categories_tokenizers}
        numeric_inputs = {feat: inputs[feat] for feat in self.normalized_feats}

        layers = []

        for feat, val in self._configs_dict["product_list"]["schema"].items():
            if "transformation" not in val.keys():
                continue
            if val["transformation"] in ["numeric", "timestamp"]:
                tensor = tf.reshape(
                    self.normalized_feats[feat](numeric_inputs[feat]), (-1, 1)
                )
                layers.append(tensor)
            # if val["transformation"] == "category":
            #     tensor = tf.one_hot(
            #         self.categories_tokenizers[feat](category_inputs[feat]),
            #         depth=self.categories_lens[feat]
            #     )
            #     layers.append(tensor)

        concatenated_inputs = tf.concat(layers, axis=-1)
        return concatenated_inputs

    def call(self, inputs):
        print("Model received  input: ", inputs)
        layers = []
        # Session features
        # layers.append(self.language(inputs.pop("language")))
        # layers.append(tf.reshape(self.latitude(inputs.pop("latitude")), (-1,1)))
        # layers.append(tf.reshape(self.longitude(inputs.pop("longitude")), (-1,1)))

        candidate_layers = self.compute_candidate_embedding(inputs)
        layers = candidate_layers

        concatenated_inputs = tf.concat(layers, axis=1)
        ratings_output = self.ratings(concatenated_inputs)

        return {"score": ratings_output}


class RankingModelModule(tfrs.models.Model):
    """
    Ranking model.
    """

    @tf.function()
    def serve(self, features):
        return self.call(features)

    def __init__(self, session_model):
        super().__init__()
        self.session_model = session_model
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, inputs):
        return self.session_model(inputs)

    def compute_loss(self, inputs, training=False):
        labels = inputs.pop("score")

        rating_predictions = self(inputs)

        # The task computes the loss and the metrics.
        return self.task(labels=labels, predictions=rating_predictions)
