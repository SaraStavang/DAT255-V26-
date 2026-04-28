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

        layers.Conv2D(16, 3,activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3,activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])


def build_deeper(input_shape):
    return keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 16,  padding='same',activation='relu'),
        layers.Conv2D(32, 16, padding='same', activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 8, padding='same', activation='relu'),
        layers.Conv2D(64, 8,  padding='same',activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 4,  padding='same',activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

def build_deepv2_model(input_shape):

    inputs = layers.Input(shape=input_shape)

    # --- Block 1 ---
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

    # --- Block 4 ---
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # --- Global pooling instead of flatten ---
    x = layers.GlobalAveragePooling2D()(x)

    # --- Dense head ---
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)

def build_deepv3_model(input_shape=(64, 64, 1)):

    inputs = layers.Input(shape=input_shape)

    # -------------------
    # Block 1
    # -------------------
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D()(x)  # 64 → 32


    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D()(x)  # 32 → 16


    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D()(x) 


    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

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



# -------------------
# RESNET50 MODEL
# -------------------
def identity_block(x, filters):
    f1, f2, f3 = filters

    shortcut = x

    x = layers.Conv2D(f1, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(f2, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(f3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

def conv_block(x, filters, stride=2):
    f1, f2, f3 = filters

    shortcut = x

    x = layers.Conv2D(f1, 1, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(f2, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(f3, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    shortcut = layers.Conv2D(f3, 1, strides=stride, padding='same')(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

def build_resnet50(input_shape=(64, 64, 1)):

    inputs = keras.Input(shape=input_shape)

    # --- Stem (lighter than original) ---
    x = layers.Conv2D(64, 3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # ======================
    # STAGE 2
    # ======================
    x = conv_block(x, [64, 64, 256], stride=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    # ======================
    # STAGE 3
    # ======================
    x = conv_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    # ======================
    # STAGE 4
    # ======================
    x = conv_block(x, [256, 256, 1024])
    for _ in range(5):
        x = identity_block(x, [256, 256, 1024])

    # ======================
    # STAGE 5
    # ======================
    x = conv_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    # ======================
    # CLASSIFICATION HEAD
    # ======================
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)


# -------------------
# RESNET18
# -------------------
def basic_block(x, filters, stride=1):

    shortcut = x

    # --- Conv 1 ---
    x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # --- Conv 2 ---
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # --- Shortcut adjust ---
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # --- Add ---
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

def build_resnet18(input_shape=(64, 64, 1)):

    inputs = layers.Input(shape=input_shape)

    # --- Initial layer (smaller than ImageNet version) ---
    x = layers.Conv2D(64, 3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # -------------------
    # Stage 1 (64 filters)
    # -------------------
    x = basic_block(x, 64)
    x = basic_block(x, 64)

    # -------------------
    # Stage 2 (128 filters)
    # -------------------
    x = basic_block(x, 128, stride=2)
    x = basic_block(x, 128)

    # -------------------
    # Stage 3 (256 filters)
    # -------------------
    x = basic_block(x, 256, stride=2)
    x = basic_block(x, 256)

    # -------------------
    # Stage 4 (512 filters)
    # -------------------
    x = basic_block(x, 512, stride=2)
    x = basic_block(x, 512)

    # -------------------
    # Head
    # -------------------
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)

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
    "deeperv2": build_deepv2_model,
    "deeperv3": build_deepv3_model,
    "FCN": build_fcn_v2,    
    "multiscale": build_multiscale_model,
    "resnet50": build_resnet50,
    "resnet18": build_resnet18,
    "mobilenetv2_transfer": build_mobilenetv2_transfer, 
}

def get_model(model_name, input_shape):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_fn = MODEL_REGISTRY[model_name]

    try:
        model = model_fn(input_shape)
    except TypeError:
        # fallback if model doesn't take input_shape
        model = model_fn()

    return model

def list_models():
    print("Available models:")
    for k in MODEL_REGISTRY.keys():
        print("-", k)
