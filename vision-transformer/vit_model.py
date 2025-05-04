"""
Derin Gezgin

File that has the vision transformer model creation.
It has a single function called create_model and it takes the parameters as a dictionary.
It constructs the model in this function and returns it for the main training script to use
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(parameter_dictionary):
    """Function to create a vision transformer model using the parameters from the parameter_dictionary"""

    TRANSFORMER_LAYER_NUMBER = parameter_dictionary["TRANSFORMER_LAYER_NUMBER"]
    EMBEDDING_DIMS = parameter_dictionary["EMBEDDING_DIMS"]
    NUM_ATTENTION_HEADS = parameter_dictionary["NUM_ATTENTION_HEADS"]
    MLP_SIZE = parameter_dictionary["MLP_SIZE"]
    NUM_CLASSES = parameter_dictionary["NUM_CLASSES"]
    DROPOUT_RATE = parameter_dictionary["DROPOUT_RATE"]
    LINEAR_LAYER_NUMBER = parameter_dictionary["LINEAR_LAYER_NUMBER"]
    NUM_PATCHES = parameter_dictionary["NUM_PATCHES"]
    PATCH_SIZE = parameter_dictionary["PATCH_SIZE"]

    class PatchEmbeddingLayer(layers.Layer):
        """This layer takes an image, divides it into patches, and converts them into a sequence of embeddings for the transformer."""

        def __init__(self, patch_size, embedding_dim):
            super().__init__()
            self.patch_size = patch_size
            self.embedding_dim = embedding_dim
            self.flatten_layer = layers.Reshape((NUM_PATCHES, embedding_dim))
            self.conv_layer = layers.Conv2D(embedding_dim, patch_size, patch_size)
            # We will initialize class_token_embeddings and position_embeddings in build()
            self.class_token_embeddings = None
            self.position_embeddings = None

        def build(self, input_shape):
            # Set the dtype for variables based on the compute dtype
            dtype = self.compute_dtype
            self.class_token_embeddings = self.add_weight(
                shape=(1, 1, self.embedding_dim),
                initializer='random_uniform',
                dtype=dtype,
                trainable=True,
                name='class_token_embeddings'
            )
            self.position_embeddings = self.add_weight(
                shape=(1, NUM_PATCHES + 1, self.embedding_dim),
                initializer='random_uniform',
                dtype=dtype,
                trainable=True,
                name='position_embeddings'
            )
            super().build(input_shape)

        def call(self, x):
            batch_size = tf.shape(x)[0]
            x = self.conv_layer(x)
            x = self.flatten_layer(x)
            class_token_embeddings = tf.tile(self.class_token_embeddings, [batch_size, 1, 1])
            x = tf.concat([class_token_embeddings, x], axis=1)
            x += self.position_embeddings
            return x

    class MultiHeadSelfAttentionBlock(layers.Layer):
        def __init__(self, embedding_dims, num_heads):
            super().__init__()
            self.norm = layers.LayerNormalization()
            self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dims)

        def call(self, x):
            x_norm = self.norm(x)
            attn_output = self.attn(x_norm, x_norm)
            return attn_output + x

    class MachineLearningPerceptronBlock(layers.Layer):
        def __init__(self, embedding_dims, mlp_size, dropout_rate):
            super().__init__()
            self.norm = layers.LayerNormalization()
            self.mlp = models.Sequential([
                layers.Dense(mlp_size, activation='gelu'),
                layers.Dropout(dropout_rate),
                layers.Dense(embedding_dims),
                layers.Dropout(dropout_rate)
            ])

        def call(self, x):
            x_norm = self.norm(x)
            return self.mlp(x_norm) + x

    class TransformerBlock(layers.Layer):
        def __init__(self, embedding_dims, num_heads, mlp_size, dropout_rate):
            super().__init__()
            self.attn_block = MultiHeadSelfAttentionBlock(embedding_dims, num_heads)
            self.mlp_block = MachineLearningPerceptronBlock(embedding_dims, mlp_size, dropout_rate)

        def call(self, x):
            x = self.attn_block(x)
            x = self.mlp_block(x)
            return x

    class VisionTransformerModel(models.Model):
        def __init__(self, num_layers, embedding_dims, num_heads, mlp_size, num_classes, dropout_rate, num_linear_layers):
            super().__init__()
            self.patch_embedding_layer = PatchEmbeddingLayer(patch_size=PATCH_SIZE, embedding_dim=EMBEDDING_DIMS)
            self.transformer_blocks = models.Sequential([TransformerBlock(embedding_dims=embedding_dims, num_heads=num_heads, mlp_size=mlp_size, dropout_rate=dropout_rate) for _ in range(num_layers)])

            # Dynamically create the MLP head with the specified number of linear layers
            mlp_layers = [layers.LayerNormalization()]
            for _ in range(num_linear_layers):
                mlp_layers.append(layers.Dense(embedding_dims, activation='gelu'))
                mlp_layers.append(layers.Dropout(dropout_rate))
            mlp_layers.append(layers.Dense(num_classes, dtype='float32'))

            self.mlp_head = models.Sequential(mlp_layers)

        def call(self, x):
            x = self.patch_embedding_layer(x)
            x = self.transformer_blocks(x)
            x = x[:, 0]
            x = self.mlp_head(x)
            return x

    ### Creating the Model ###
    vision_transformer_model = VisionTransformerModel(
        num_layers=TRANSFORMER_LAYER_NUMBER,
        embedding_dims=EMBEDDING_DIMS,
        num_heads=NUM_ATTENTION_HEADS,
        mlp_size=MLP_SIZE,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        num_linear_layers=LINEAR_LAYER_NUMBER,
    )

    return vision_transformer_model
