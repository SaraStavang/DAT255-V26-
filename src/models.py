import keras
from keras import layers
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input


# -------------------
# MODELS
# -------------------
def build_baseline(input_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])


def build_deeper(input_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 32, activation='relu'),
        layers.Conv2D(32, 32, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 32, activation='relu'),
        layers.Conv2D(64, 32, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 16, activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

def build_stronger_model(input_shape):

    inputs = layers.Input(shape=input_shape)

    # --- Block 1 (bigger filters early) ---
    x = layers.Conv2D(32, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # --- Block 2 ---
    x = layers.Conv2D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # --- Block 3 ---
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    # --- Block 4 (important for context) ---
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # --- Global pooling instead of flatten ---
    x = layers.GlobalAveragePooling2D()(x)

    # --- Dense head ---
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)

def build_fcn_v2(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(128, 1, activation='relu')(x)

    # classification map
    x = layers.Conv2D(1, 1, activation='sigmoid')(x)

    # reduce to scalar
    x = layers.GlobalMaxPooling2D()(x)

    return keras.Model(inputs, x)

def residual_block(x, filters, stride=1):

    shortcut = x

    # main path
    x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # adjust shortcut if needed
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


def build_deep_resnet_fcn(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # --- Stage 1 ---
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    # --- Stage 2 ---
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)

    # --- Stage 3 ---
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)

    # --- Stage 4 ---
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    # --- FCN head ---
    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(1, 1, activation='sigmoid')(x)

    # --- global decision ---
    x = layers.GlobalMaxPooling2D()(x)

    return keras.Model(inputs, x)

def build_patch_cnn(input_shape=(64, 64, 1)):

    inputs = keras.Input(shape=input_shape)

    # --- Block 1 ---
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D()(x) 

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D()(x)  

  
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D()(x) 


    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)

def build_multiscale_model(input_shape):

    # --- INPUTS ---
    inp1 = keras.Input(shape=input_shape)  # small
    inp2 = keras.Input(shape=input_shape)  # large (downsampled)

    # --- SHARED FEATURE EXTRACTOR ---
    def branch(x):
        x = layers.Conv2D(32, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.GlobalAveragePooling2D()(x)

        return x

    # --- APPLY TO BOTH SCALES ---
    f1 = branch(inp1)
    f2 = branch(inp2)

    # --- MERGE ---
    x = layers.Concatenate()([f1, f2])

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model([inp1, inp2], out)

def build_mobilenetv2_transfer(input_shape):
    inputs = layers.Input(shape=input_shape)

    # grayscale -> RGB
    x = layers.Concatenate()([inputs, inputs, inputs])

    # MobileNetV2 preprocess
    x = layers.Rescaling(255.0)(x)
    x = layers.Lambda(preprocess_input)(x)

    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(input_shape[0], input_shape[1], 3)
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)
    
MODEL_REGISTRY = {
    "baseline": build_baseline,
    "deeper": build_deeper,
    "strong": build_stronger_model,
    "FCN": build_fcn_v2,
    "DFCN": build_deep_resnet_fcn,
    "PatchCNN": build_patch_cnn,
    "multiscale": build_multiscale_model,
    "mobilenetv2_transfer": build_mobilenetv2_transfer, 
}

def load_model(model_name, input_shape):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_fn = MODEL_REGISTRY[model_name]

    try:
        model = model_fn(input_shape)
    except TypeError:
        # fallback if model doesn't take input_shape (rare case)
        model = model_fn()

    return model

def list_models():
    print("Available models:")
    for k in MODEL_REGISTRY.keys():
        print("-", k)
