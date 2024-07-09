import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import StringLookup
import matplotlib.pyplot as plt
import pickle
import re


data_dir = "/content/data"
word_entries = []
with open(f"{data_dir}/words.txt", "r") as file:
    for line in file:
        if line[0] != "#" and line.split(" ")[1] != "err":
            word_entries.append(line.strip())

np.random.shuffle(word_entries)

split_index = int(0.8 * len(word_entries))
train_entries = word_entries[:split_index]
remaining_entries = word_entries[split_index:]
val_index = int(0.5 * len(remaining_entries))
val_entries = remaining_entries[:val_index]
test_entries = remaining_entries[val_index:]

assert len(word_entries) == len(train_entries) + len(val_entries) + len(test_entries)

print(f"Total Training Samples: {len(train_entries)}")
print(f"Total Validation Samples: {len(val_entries)}")
print(f"Total Test Samples: {len(test_entries)}")

image_base_path = os.path.join(data_dir, "words")

def extract_image_paths_and_labels(samples):
    img_paths, corrected_labels = [], []
    for line in samples:
        parts = line.split(" ")
        img_name = parts[0]
        part1, part2 = img_name.split("-")[:2]
        img_path = os.path.join(image_base_path, part1, f"{part1}-{part2}", f"{img_name}.png")
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            img_paths.append(img_path)
            corrected_labels.append(line)
    return img_paths, corrected_labels

train_img_paths, train_labels = extract_image_paths_and_labels(train_entries)
val_img_paths, val_labels = extract_image_paths_and_labels(val_entries)
test_img_paths, test_labels = extract_image_paths_and_labels(test_entries)

unique_chars = set()
max_label_len = 0
cleaned_train_labels = []

for label in train_labels:
    word = label.split(" ")[-1]
    unique_chars.update(word)
    max_label_len = max(max_label_len, len(word))
    cleaned_train_labels.append(word)

def clean_labels(labels):
    return [label.split(" ")[-1] for label in labels]

cleaned_val_labels = clean_labels(val_labels)
cleaned_test_labels = clean_labels(test_labels)

with open("/content/drive/MyDrive/Colab Notebooks/characters", "rb") as file:
    char_mapping = pickle.load(file)

char_to_num = StringLookup(vocabulary=char_mapping, mask_token=None)
num_to_chars = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

"""**Preprocessing**"""

def resize_and_pad_image(image, img_size):
    width, height = img_size
    image = tf.image.resize(image, size=(height, width), preserve_aspect_ratio=True)
    pad_height = height - tf.shape(image)[0]
    pad_width = width - tf.shape(image)[1]

    pad_height_top = pad_height // 2
    pad_height_bottom = pad_height - pad_height_top
    pad_width_left = pad_width // 2
    pad_width_right = pad_width - pad_width_left

    image = tf.pad(image, paddings=[[pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]])
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

img_width, img_height = 128, 32
batch_size = 64
padding_token = 99

def preprocess_image(image_path, img_size=(img_width, img_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = resize_and_pad_image(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    label_length = tf.shape(label)[0]
    padding = max_label_len - label_length
    label = tf.pad(label, paddings=[[0, padding]], constant_values=padding_token)
    return label

def preprocess_images_and_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}

def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(preprocess_images_and_labels, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

train_dataset = prepare_dataset(train_img_paths, cleaned_train_labels)
val_dataset = prepare_dataset(val_img_paths, cleaned_val_labels)
test_dataset = prepare_dataset(test_img_paths, cleaned_test_labels)

"""**Visualization**"""

def visualize_samples(dataset, save_path, num_samples=16):
    for batch in dataset.take(1):
        images, labels = batch["image"], batch["label"]
        fig, ax = plt.subplots(4, 4, figsize=(15, 8))
        for i in range(num_samples):
            img = tf.image.flip_left_right(tf.transpose(images[i], perm=[1, 0, 2]))
            img = (img * 255.0).numpy().astype(np.uint8)[:, :, 0]
            label = labels[i]
            decoded_label = tf.strings.reduce_join(num_to_chars(tf.gather(label, tf.where(tf.math.not_equal(label, padding_token)))))
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(decoded_label.numpy().decode("utf-8"))
            ax[i // 4, i % 4].axis("off")
        plt.savefig(save_path)

visualize_samples(train_dataset, "train_samples.png")
visualize_samples(test_dataset, "test_samples.png")

"""**Model Definition**"""

class CustomCTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_size = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones(shape=(batch_size, 1), dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones(shape=(batch_size, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred
def create_model():
    input_img = keras.Input(shape=(img_width, img_height, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", name="Conv1")(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv2")(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    output = keras.layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)
    output = CustomCTCLayer(name="ctc_loss")(labels, output)

    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer)
    return model
model = create_model()
model.summary()

"""**Model Training**"""

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,
    callbacks=[early_stopping_cb]
)

def plot_training_history(history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)

plot_training_history(history, "training_history.png")

"""**Inference**"""

custom_objects = {"CustomCTCLayer": CustomCTCLayer}
reconstructed_model = keras.models.load_model("/content/drive/MyDrive/Colab Notebooks/saved_pred_model.h5", custom_objects=custom_objects)
prediction_model = keras.models.Model(
    inputs=reconstructed_model.get_layer(name="image").input,
    outputs=reconstructed_model.get_layer(name="dense2").output
)

pred_test_text = []

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_label_len]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_chars(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

for batch in test_dataset.take(3):
    batch_images = batch["image"]
    print(batch_images.shape)
    _, ax = plt.subplots(4, 4, figsize=(15, 8))
    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)
    pred_test_text.append(pred_texts)
    for i in range(16):
        img = batch_images[i]
        img = tf.image.flip_left_right(img)
        img = tf.transpose(img, perm=[1, 0, 2])
        img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        img = img[:, :, 0]

        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
    plt.savefig("Inference1.png")
flat_list = [item for sublist in pred_test_text for item in sublist]
print(flat_list)
sentence = ' '.join(flat_list)
print(sentence)

