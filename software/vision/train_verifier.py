import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATASET_DIR = "dataset"
MODEL_OUT = "micro_verify_net.keras"

def load_data(dataset_dir):
    balls_paths = glob.glob(os.path.join(dataset_dir, "balls", "*.png"))
    distractors_paths = glob.glob(os.path.join(dataset_dir, "distractors", "*.png"))
    
    X = []
    Y = []
    
    print(f"Found {len(balls_paths)} balls and {len(distractors_paths)} distractors.")
    
    def process_img(path, label):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return
        if img_bgr.shape[:2] != (32, 32):
            img_bgr = cv2.resize(img_bgr, (32, 32))
            
        # Convert to YUV (Critical for luminance shading highlights/shadows!)
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        
        # Normalize to 0.0 - 1.0
        arr = img_yuv.astype(np.float32) / 255.0
        X.append(arr)
        Y.append(label)
        
    for p in balls_paths:
        process_img(p, 1.0)
    for p in distractors_paths:
        process_img(p, 0.0)
        
    return np.array(X), np.array(Y)

def dw_sep_block(x, filters):
    x = layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_model():
    inputs = layers.Input(shape=(32, 32, 3), name="input_yuv")
    
    # Data Augmentation
    # NOTE: We ONLY allow horizontal flip! 
    # Vertical flip destroys the shading physics (bright highlight on top, shadow on bottom).
    x = layers.RandomFlip("horizontal")(inputs)
    
    x = layers.Conv2D(16, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x) # Down to 16x16
    
    x = dw_sep_block(x, 32)
    x = layers.MaxPooling2D()(x) # Down to 8x8
    
    x = dw_sep_block(x, 64)
    
    # Classification Head
    gap = layers.GlobalAveragePooling2D(name="gap")(x)
    dense = layers.Dense(32, use_bias=False)(gap)
    dense = layers.BatchNormalization()(dense)
    dense = layers.ReLU()(dense)
    dense = layers.Dropout(0.3)(dense)
    
    # Output is a single probability (0 = Distractor, 1 = Ball)
    out_detect = layers.Dense(1, activation='sigmoid', name='is_ball')(dense)
    
    model = models.Model(inputs=inputs, outputs=out_detect)
    return model

def main():
    X, Y = load_data(DATASET_DIR)
    
    if len(X) == 0:
        print("Dataset is empty. Run extract_candidates.py first and put images in dataset/balls and dataset/distractors!")
        return
        
    # Shuffle dataset
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    
    # 80/20 Train/Val Split
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split], Y[:split]
    X_val, Y_val = X[split:], Y[split:]
    
    model = build_model()
    model.summary()
    print(f"\nTotal Parameters: {model.count_params():,}")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    
    print("\n--- Training MicroVerifyNet ---")
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr]
    )
    
    model.save(MODEL_OUT)
    print(f"\nKeras Model saved to {MODEL_OUT}")
    
    # ── TFLITE INT8 CONVERSION ──
    print("\n--- Converting to TFLite INT8 ---")
    
    def representative_dataset():
        sample_idx = np.random.choice(len(X_train), size=min(200, len(X_train)), replace=False)
        for i in sample_idx:
            yield [np.expand_dims(X_train[i], axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Keep input/output as float32 for seamless Python integration, 
    # but internally everything is quantized to INT8!
    converter.inference_input_type = tf.float32 
    converter.inference_output_type = tf.float32
    
    tflite_model = converter.convert()
    tflite_path = MODEL_OUT.replace(".keras", "_int8.tflite")
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"\nTFLite INT8 successfully exported to: {tflite_path} ({len(tflite_model)/1024:.1f} KB)")
    print("You can now integrate this into color_tracking.py!")

if __name__ == "__main__":
    main()
