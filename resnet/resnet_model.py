import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(parameter_dictionary):

    NUM_CLASSES = parameter_dictionary.get("NUM_CLASSES", 2)
    INITIAL_FILTERS = parameter_dictionary.get("INITIAL_FILTERS", 64)
    BLOCKS_PER_STAGE = parameter_dictionary.get("BLOCKS_PER_STAGE", [3, 4, 6, 3])
    FILTER_MULTIPLIER_PER_STAGE = parameter_dictionary.get("FILTER_MULTIPLIER_PER_STAGE", [1, 2, 4, 8])
    KERNEL_SIZE = parameter_dictionary.get("KERNEL_SIZE", 3)
    STRIDES_PER_STAGE = parameter_dictionary.get("STRIDES_PER_STAGE", [1, 2, 2, 2])
    DROPOUT_RATE = parameter_dictionary.get("DROPOUT_RATE", 0.2)
    LINEAR_LAYER_NUMBER = parameter_dictionary.get("LINEAR_LAYER_NUMBER", 3)
    IMAGE_HEIGHT = parameter_dictionary.get("IMAGE_HEIGHT", 600)
    IMAGE_WIDTH = parameter_dictionary.get("IMAGE_WIDTH", 200)
    IMAGE_CHANNELS = parameter_dictionary.get("IMAGE_CHANNELS", 3)

    # Define a residual block
    class ResidualBlock(layers.Layer):
        def __init__(self, filters, kernel_size=3, stride=1):
            super().__init__()
            self.conv1 = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)
            self.bn1 = layers.BatchNormalization()
            self.act1 = layers.ReLU()

            self.conv2 = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False)
            self.bn2 = layers.BatchNormalization()

            self.shortcut = lambda x: x
            if stride != 1:
                # Projection shortcut to match dimensions
                self.shortcut = models.Sequential([
                    layers.Conv2D(filters, 1, strides=stride, use_bias=False),
                    layers.BatchNormalization()
                ])
            
        def call(self, x):
            shortcut = x
            # First conv
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)

            # Second conv
            x = self.conv2(x)
            x = self.bn2(x)

            # Add shortcut
            shortcut = self.shortcut(shortcut)
            x = layers.add([x, shortcut])
            x = tf.nn.relu(x)
            return x

    # Define a stage of multiple residual blocks
    class ResNetStage(layers.Layer):
        def __init__(self, filters, num_blocks, kernel_size=3, stride=1):
            super().__init__()
            self.blocks = []
            # First block may need to adjust stride for downsampling
            self.blocks.append(ResidualBlock(filters, kernel_size=kernel_size, stride=stride))
            # Remaining blocks have stride 1
            for _ in range(num_blocks - 1):
                self.blocks.append(ResidualBlock(filters, kernel_size=kernel_size, stride=1))

        def call(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    # Define the full ResNet model
    class ResNetModel(models.Model):
        def __init__(self, 
                     num_classes,
                     initial_filters,
                     blocks_per_stage,
                     filter_multipliers,
                     kernel_size,
                     strides_per_stage,
                     dropout_rate,
                     linear_layer_number):
            super().__init__()

            self.stem = models.Sequential([
                layers.Conv2D(initial_filters, 7, strides=2, padding='same', use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
            ])

            self.stages = []
            for i, (num_blocks, stride, mult) in enumerate(zip(blocks_per_stage, strides_per_stage, filter_multipliers)):
                filters = initial_filters * mult
                self.stages.append(ResNetStage(filters=filters, 
                                               num_blocks=num_blocks, 
                                               kernel_size=kernel_size, 
                                               stride=stride))

            self.global_pool = layers.GlobalAveragePooling2D()

            # Create an MLP head similar to ViT code
            mlp_layers = []
            for _ in range(linear_layer_number):
                mlp_layers.append(layers.Dense(initial_filters * filter_multipliers[-1], activation='relu'))
                mlp_layers.append(layers.Dropout(dropout_rate))
            mlp_layers.append(layers.Dense(num_classes, dtype='float32'))

            self.classifier = models.Sequential(mlp_layers)

        def call(self, x):
            x = self.stem(x)
            for stage in self.stages:
                x = stage(x)
            x = self.global_pool(x)
            x = self.classifier(x)
            return x

    # Create the model
    resnet_model = ResNetModel(
        num_classes=NUM_CLASSES,
        initial_filters=INITIAL_FILTERS,
        blocks_per_stage=BLOCKS_PER_STAGE,
        filter_multipliers=FILTER_MULTIPLIER_PER_STAGE,
        kernel_size=KERNEL_SIZE,
        strides_per_stage=STRIDES_PER_STAGE,
        dropout_rate=DROPOUT_RATE,
        linear_layer_number=LINEAR_LAYER_NUMBER
    )

    return resnet_model

