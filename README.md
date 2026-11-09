<details>
<summary>ENG (English Version)</summary>

# Week 05 — Variational Autoencoder and Image Generation

Implementation of a convolutional variational autoencoder (VAE) with TensorFlow and Keras. Reconstruction and generation of MNIST digit images, followed by an optional face generation exercise with RGB images.

## 1. Project Overview

Extension of the earlier MNIST work from digit classification to image reconstruction and generation.

| Stage | Implementation |
| --- | --- |
| Encoding | Estimation of the latent mean and log variance for each input |
| Sampling | Construction of a latent vector through reparameterization |
| Reconstruction | Decoding of a latent representation obtained from an input image |
| Generation | Decoding of a random vector drawn from the standard normal prior |
| Application | Training of a separate VAE for `64 × 64` RGB face images |

---

## 2. Autoencoder and Variational Autoencoder

| Category | Standard Autoencoder (AE) | Variational Autoencoder (VAE) |
| --- | --- | --- |
| Latent representation | One deterministic vector per input | A probability distribution per input |
| Encoder output | Latent vector `z` | Mean `z_mean` and log variance `z_log_var` |
| Training objective in this comparison | Reconstruction loss | Reconstruction loss and KL divergence |
| Use in this exercise | Image compression and reconstruction | Image reconstruction, prior sampling, and latent interpolation |

```text
AE  : Input → Encoder → Latent vector → Decoder → Reconstruction
VAE : Input → Encoder → Mean and log variance → Sampling → Decoder → Reconstruction
```

---

## 3. Reparameterization and Training Loss

Sampling of standard normal noise `epsilon` and transformation with the encoder's mean and standard deviation.

```text
z_std = exp(0.5 * z_log_var)
z = z_mean + z_std * epsilon
```

Separation of random noise from the encoder outputs, allowing gradients to flow through `z_mean` and `z_log_var`.

| Loss term | Calculation in this exercise |
| --- | --- |
| Reconstruction loss | Sum of elementwise binary cross-entropy across image dimensions and channels |
| KL divergence | Divergence from the input-specific latent distribution `q(z \| x)` to the standard normal prior `p(z) = N(0, I)` |
| Total loss | Batch mean of the sum of both terms for each image |

---

## 4. MNIST Digit Reconstruction and Generation

### 4.1. Practice Setup

```bash
python -m pip install "tensorflow>=2.16" matplotlib
```

| Setting | Value |
| --- | --- |
| Input shape | `(28, 28, 1)` |
| Pixel range | `[0, 1]` |
| Latent dimensions | `2` |
| Optimizer | Adam |
| Training | 30 epochs, batch size 128 |
| Test images | Five images reserved for reconstruction comparison |

### 4.2. Model Implementation and Training

Creation of `train_mnist_vae.py` in the project directory.

```python
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        noise = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + tf.exp(0.5 * log_var) * noise


def build_models(image_shape, latent_dim):
    height, width, channels = image_shape
    feature_shape = (height // 4, width // 4, 64)

    image = keras.Input(shape=image_shape)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(image)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Dense(16, activation="relu")(layers.Flatten()(x))
    mean = layers.Dense(latent_dim, name="z_mean")(x)
    log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    latent = Sampling()([mean, log_var])
    encoder = keras.Model(image, [mean, log_var, latent], name="encoder")

    decoder = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(feature_shape[0] * feature_shape[1] * 64, activation="relu"),
            layers.Reshape(feature_shape),
            layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(channels, 3, padding="same", activation="sigmoid"),
        ],
        name="decoder",
    )
    return encoder, decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_trackers = [
            keras.metrics.Mean(name=name)
            for name in ("loss", "reconstruction_loss", "kl_loss")
        ]

    @property
    def metrics(self):
        return self.loss_trackers

    def train_step(self, data):
        images, _, _ = keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            mean, log_var, latent = self.encoder(images, training=True)
            reconstructed = self.decoder(latent, training=True)

            pixel_bce = keras.ops.binary_crossentropy(images, reconstructed)
            reconstruction = tf.reduce_sum(pixel_bce, axis=(1, 2, 3))
            kl = 0.5 * tf.reduce_sum(
                tf.square(mean) + tf.exp(log_var) - 1.0 - log_var,
                axis=1,
            )
            total = reconstruction + kl
            loss = tf.reduce_mean(total)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        for tracker, values in zip(self.metrics, (total, reconstruction, kl)):
            tracker.update_state(values)

        return {tracker.name: tracker.result() for tracker in self.metrics}


def save_grid(rows, filename):
    columns = len(rows[0])
    fig, axes = plt.subplots(
        len(rows), columns, squeeze=False, figsize=(2 * columns, 2 * len(rows))
    )

    for row_axes, images in zip(axes, rows):
        for axis, image in zip(row_axes, images):
            if image.shape[-1] == 1:
                axis.imshow(image[..., 0], cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(image)
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    keras.utils.set_random_seed(42)
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train[..., None].astype("float32") / 255.0
    originals = x_test[:5, ..., None].astype("float32") / 255.0

    latent_dim = 2
    encoder, decoder = build_models((28, 28, 1), latent_dim)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(x_train, epochs=30, batch_size=128)

    means, _, _ = encoder(originals, training=False)
    reconstructed = decoder(means, training=False).numpy()
    save_grid([originals, reconstructed], "mnist_reconstructions.png")

    samples = tf.random.normal((5, latent_dim))
    generated = decoder(samples, training=False).numpy()
    save_grid([generated], "mnist_generated.png")


if __name__ == "__main__":
    main()
```

### 4.3. Execution and Output Inspection

```bash
python train_mnist_vae.py
```

| Output | Inspection |
| --- | --- |
| `mnist_reconstructions.png` | Comparison of original test digits in the top row with reconstructions in the bottom row |
| `mnist_generated.png` | Five digit images decoded from independently sampled prior vectors |
| Training log | Tracking of `loss`, `reconstruction_loss`, and `kl_loss` across epochs |

---

## 5. Application — Face Image Generation

Reuse of the CNN builder and training loop with a separate model for `64 × 64` RGB images and a 200-dimensional latent space. Placement of face images in `data/faces/` and storage of `train_faces_vae.py` beside the MNIST script.

```python
import tensorflow as tf
from tensorflow import keras

from train_mnist_vae import VAE, build_models, save_grid


Z_DIM = 200


class ImageGenerator(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.latents = tf.random.normal((5, Z_DIM))

    def on_epoch_end(self, epoch, logs=None):
        images = self.model.decoder(self.latents, training=False).numpy()
        save_grid([images], f"faces_epoch_{epoch + 1:02d}.png")


def main():
    train = keras.utils.image_dataset_from_directory(
        "data/faces",
        labels=None,
        color_mode="rgb",
        image_size=(64, 64),
        batch_size=64,
    )
    train = train.map(lambda images: images / 255.0).prefetch(tf.data.AUTOTUNE)

    encoder, decoder = build_models((64, 64, 3), Z_DIM)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))
    vae.fit(train, epochs=20, callbacks=[ImageGenerator()])


if __name__ == "__main__":
    main()
```

```bash
python train_faces_vae.py
```

- **Epoch comparison** — Decoding of the same five latent vectors after every epoch and comparison of the `faces_epoch_*.png` grids.
- **Latent interpolation** — Linear interpolation between the encoded means of two input faces and decoding of the intermediate vectors to inspect the transition.


</details>

<details>
<summary>KOR (한국어 버전)</summary>

# 5주차 — 변분 오토인코더와 이미지 생성

TensorFlow와 Keras를 활용한 합성곱 기반 변분 오토인코더(VAE) 구현. MNIST 손글씨 이미지 복원 및 생성과 RGB 얼굴 이미지를 활용한 선택 실습.

## 1. 프로젝트 개요

앞선 MNIST 실습의 손글씨 숫자 분류에서 이미지 복원 및 생성으로의 확장.

| 구분 | 구현 내용 |
| --- | --- |
| 인코딩 | 입력 이미지별 잠재 분포의 평균과 로그 분산 추정 |
| 샘플링 | 재매개변수화를 통한 잠재 벡터 구성 |
| 복원 | 입력 이미지에서 얻은 잠재 표현을 디코딩 |
| 생성 | 표준 정규 사전분포에서 뽑은 잠재 벡터를 디코딩 |
| 응용 | `64 × 64` RGB 얼굴 이미지용 VAE 별도 학습 |

---

## 2. 오토인코더와 변분 오토인코더

| 구분 | 일반 오토인코더(AE) | 변분 오토인코더(VAE) |
| --- | --- | --- |
| 잠재 표현 | 입력마다 하나의 고정 잠재 벡터 | 입력마다 하나의 잠재 확률분포 |
| 인코더 출력 | 잠재 벡터 `z` | 평균 `z_mean`, 로그 분산 `z_log_var` |
| 비교에 사용한 학습 목표 | 재구성 손실 | 재구성 손실과 KL 발산 |
| 실습 활용 | 이미지 압축 및 복원 | 이미지 복원, 사전분포 샘플링, 잠재 공간 보간 |

```text
AE  : 입력 → 인코더 → 잠재 벡터 → 디코더 → 복원
VAE : 입력 → 인코더 → 평균과 로그 분산 → 샘플링 → 디코더 → 복원
```

---

## 3. 재매개변수화와 학습 손실

표준 정규분포에서 잡음 `epsilon`을 뽑고 인코더에서 계산한 평균과 표준편차를 적용해 잠재 벡터 구성.

```text
z_std = exp(0.5 * z_log_var)
z = z_mean + z_std * epsilon
```

무작위 잡음을 인코더 출력과 분리해 `z_mean`과 `z_log_var`를 거쳐 인코더로 기울기 전달.

| 손실 항목 | 실습에서의 계산 방법 |
| --- | --- |
| 재구성 손실 | 이미지의 높이·너비·채널별 이진 교차 엔트로피(BCE) 합산 |
| KL 발산 | 입력별 잠재 분포 `q(z \| x)`에서 표준 정규 사전분포 `p(z) = N(0, I)`로의 KL 발산 계산 |
| 전체 손실 | 이미지별 재구성 손실과 KL 발산을 더한 뒤 배치 평균 계산 |

---

## 4. MNIST 손글씨 이미지 복원 및 생성

### 4.1. 실습 환경 구성

```bash
python -m pip install "tensorflow>=2.16" matplotlib
```

| 구분 | 설정 |
| --- | --- |
| 입력 형태 | `(28, 28, 1)` |
| 픽셀 범위 | `[0, 1]` |
| 잠재 차원 | `2` |
| 옵티마이저 | Adam |
| 학습 설정 | 30 에포크, 배치 크기 128 |
| 테스트 이미지 | 복원 비교에 사용할 테스트 이미지 5장 |

### 4.2. 모델 구현 및 학습

프로젝트 폴더에 `train_mnist_vae.py` 작성.

```python
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        noise = tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        return mean + tf.exp(0.5 * log_var) * noise


def build_models(image_shape, latent_dim):
    height, width, channels = image_shape
    feature_shape = (height // 4, width // 4, 64)

    image = keras.Input(shape=image_shape)
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(image)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Dense(16, activation="relu")(layers.Flatten()(x))
    mean = layers.Dense(latent_dim, name="z_mean")(x)
    log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    latent = Sampling()([mean, log_var])
    encoder = keras.Model(image, [mean, log_var, latent], name="encoder")

    decoder = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            layers.Dense(feature_shape[0] * feature_shape[1] * 64, activation="relu"),
            layers.Reshape(feature_shape),
            layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu"),
            layers.Conv2D(channels, 3, padding="same", activation="sigmoid"),
        ],
        name="decoder",
    )
    return encoder, decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_trackers = [
            keras.metrics.Mean(name=name)
            for name in ("loss", "reconstruction_loss", "kl_loss")
        ]

    @property
    def metrics(self):
        return self.loss_trackers

    def train_step(self, data):
        images, _, _ = keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            mean, log_var, latent = self.encoder(images, training=True)
            reconstructed = self.decoder(latent, training=True)

            pixel_bce = keras.ops.binary_crossentropy(images, reconstructed)
            reconstruction = tf.reduce_sum(pixel_bce, axis=(1, 2, 3))
            kl = 0.5 * tf.reduce_sum(
                tf.square(mean) + tf.exp(log_var) - 1.0 - log_var,
                axis=1,
            )
            total = reconstruction + kl
            loss = tf.reduce_mean(total)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        for tracker, values in zip(self.metrics, (total, reconstruction, kl)):
            tracker.update_state(values)

        return {tracker.name: tracker.result() for tracker in self.metrics}


def save_grid(rows, filename):
    columns = len(rows[0])
    fig, axes = plt.subplots(
        len(rows), columns, squeeze=False, figsize=(2 * columns, 2 * len(rows))
    )

    for row_axes, images in zip(axes, rows):
        for axis, image in zip(row_axes, images):
            if image.shape[-1] == 1:
                axis.imshow(image[..., 0], cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(image)
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    keras.utils.set_random_seed(42)
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train[..., None].astype("float32") / 255.0
    originals = x_test[:5, ..., None].astype("float32") / 255.0

    latent_dim = 2
    encoder, decoder = build_models((28, 28, 1), latent_dim)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(x_train, epochs=30, batch_size=128)

    means, _, _ = encoder(originals, training=False)
    reconstructed = decoder(means, training=False).numpy()
    save_grid([originals, reconstructed], "mnist_reconstructions.png")

    samples = tf.random.normal((5, latent_dim))
    generated = decoder(samples, training=False).numpy()
    save_grid([generated], "mnist_generated.png")


if __name__ == "__main__":
    main()
```

### 4.3. 실행 및 결과 확인

```bash
python train_mnist_vae.py
```

| 출력 | 확인 내용 |
| --- | --- |
| `mnist_reconstructions.png` | 첫째 줄의 테스트 원본과 둘째 줄의 복원 이미지 비교 |
| `mnist_generated.png` | 사전분포에서 각각 뽑은 잠재 벡터로 생성한 손글씨 이미지 5장 확인 |
| 학습 로그 | 에포크별 `loss`, `reconstruction_loss`, `kl_loss` 변화 확인 |

---

## 5. 응용 실습 — 얼굴 이미지 생성

MNIST의 CNN 구성 함수와 학습 코드 재사용. `64 × 64` RGB 입력과 200차원 잠재 공간에 맞춘 별도 모델 구성. 얼굴 이미지 파일은 `data/faces/`에 배치하고, `train_faces_vae.py`는 MNIST 스크립트와 같은 폴더에 작성.

```python
import tensorflow as tf
from tensorflow import keras

from train_mnist_vae import VAE, build_models, save_grid


Z_DIM = 200


class ImageGenerator(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.latents = tf.random.normal((5, Z_DIM))

    def on_epoch_end(self, epoch, logs=None):
        images = self.model.decoder(self.latents, training=False).numpy()
        save_grid([images], f"faces_epoch_{epoch + 1:02d}.png")


def main():
    train = keras.utils.image_dataset_from_directory(
        "data/faces",
        labels=None,
        color_mode="rgb",
        image_size=(64, 64),
        batch_size=64,
    )
    train = train.map(lambda images: images / 255.0).prefetch(tf.data.AUTOTUNE)

    encoder, decoder = build_models((64, 64, 3), Z_DIM)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005))
    vae.fit(train, epochs=20, callbacks=[ImageGenerator()])


if __name__ == "__main__":
    main()
```

```bash
python train_faces_vae.py
```

- **에포크별 비교** — 매 에포크 종료 후 같은 잠재 벡터 5개를 디코딩하고 `faces_epoch_*.png` 그리드 비교.
- **잠재 공간 보간** — 두 입력 얼굴의 잠재 평균 벡터 사이를 선형 보간한 뒤, 중간 벡터를 디코딩해 두 얼굴 사이의 변화 확인.


</details>
