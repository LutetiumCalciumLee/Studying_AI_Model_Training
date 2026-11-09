<details>
<summary>ENG (English Version)</summary>

# Week 06 — Generative Adversarial Networks and Image Generation

Implementation of a convolutional generative adversarial network (GAN) with TensorFlow and Keras. Generation of Quick, Draw! camel sketches and MNIST digits, with inspection of generated samples and training losses.

## 1. GAN Architecture

Adversarial training of two neural networks to learn a data distribution and generate new samples.

| Component | Input | Role |
| --- | --- | --- |
| Generator (G) | A 100-dimensional vector `z` sampled from a standard normal distribution | Generation of a `28 × 28 × 1` grayscale image |
| Discriminator (D) | A real image or an image produced by G | Prediction of a binary classification logit for real versus generated images |

```text
Random noise z → Generator → Generated image ─┐
                                              ├→ Discriminator → Real/fake score
Training dataset ──────────→ Real image ───────┘
```

---

## 2. Alternating Training

| Update | Training targets | Parameters updated |
| --- | --- | --- |
| Discriminator | Real images → `1`, generated images → `0` | Discriminator only |
| Generator | Generated images passed through the discriminator → target `1` | Generator only |

Binary cross-entropy for both updates. During the generator update, backpropagation through the discriminator's output to the generator, with the discriminator weights held fixed.

---

## 3. Quick, Draw! Camel Sketch Generation

### 3.1. Dataset and Configuration

| Item | Setting |
| --- | --- |
| Training data | First 10,000 sketches from Google's Quick, Draw! `camel` class |
| Input preprocessing | Reshape to `(N, 28, 28, 1)` and rescale pixel values from `[0, 255]` to `[-1, 1]` |
| Generator | Projection to `7 × 7 × 64`, followed by two transposed convolutions and a `tanh` output |
| Discriminator | Strided convolutions, Leaky ReLU, dropout, and a single output logit |
| Optimizers | RMSprop with `0.0008` for D and `0.0004` for G |
| Training | Batch size `64` and `2` epochs for an introductory run |

### 3.2. Setup and Execution

Installation of dependencies and download of the official NumPy bitmap file under the name expected by the script.

```bash
python -m pip install "tensorflow>=2.16" numpy matplotlib
curl -fL https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/camel.npy -o full_numpy_bitmap_camel.npy
```

Execution after saving the code below as `train_gan.py` in the same directory as the data file.

```bash
python train_gan.py
```

### 3.3. Training and Visualization Code

```python
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = ArgumentParser()
parser.add_argument("--mnist", action="store_true")
args = parser.parse_args()
keras.utils.set_random_seed(42)

if args.mnist:
    (raw, _), _ = keras.datasets.mnist.load_data()
else:
    raw = np.load("full_numpy_bitmap_camel.npy", mmap_mode="r")

images = np.asarray(raw[:10_000], dtype=np.float32).reshape(-1, 28, 28, 1)
images = images / 127.5 - 1.0
batches = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(64)

generator = keras.Sequential([
    keras.Input(shape=(100,)),
    layers.Dense(7 * 7 * 64, activation="relu"),
    layers.Reshape((7, 7, 64)),
    layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu"),
    layers.Conv2DTranspose(1, 4, strides=2, padding="same", activation="tanh"),
])
discriminator = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(64, 5, strides=2, padding="same"),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Conv2D(128, 5, strides=2, padding="same"),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(1),
])

bce = keras.losses.BinaryCrossentropy(from_logits=True)
d_optimizer = keras.optimizers.RMSprop(learning_rate=0.0008)
g_optimizer = keras.optimizers.RMSprop(learning_rate=0.0004)
d_history, g_history = [], []
fixed_noise = tf.random.normal((16, 100))

for epoch in range(1, 3):
    for real in batches:
        noise_shape = (tf.shape(real)[0], 100)
        fake = generator(tf.random.normal(noise_shape), training=False)

        with tf.GradientTape() as tape:
            real_scores = discriminator(real, training=True)
            fake_scores = discriminator(fake, training=True)
            d_loss = (
                bce(tf.ones_like(real_scores), real_scores)
                + bce(tf.zeros_like(fake_scores), fake_scores)
            ) / 2
        d_weights = discriminator.trainable_weights
        d_optimizer.apply_gradients(zip(tape.gradient(d_loss, d_weights), d_weights))

        with tf.GradientTape() as tape:
            fake = generator(tf.random.normal(noise_shape), training=True)
            fake_scores = discriminator(fake, training=False)
            g_loss = bce(tf.ones_like(fake_scores), fake_scores)
        g_weights = generator.trainable_weights
        g_optimizer.apply_gradients(zip(tape.gradient(g_loss, g_weights), g_weights))

        d_history.append(float(d_loss))
        g_history.append(float(g_loss))

    samples = (generator(fixed_noise, training=False).numpy() + 1.0) / 2.0
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for axis, sample in zip(axes.flat, samples):
        axis.imshow(sample[..., 0], cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(f"samples_epoch_{epoch}.png")
    plt.close(fig)

fig, axis = plt.subplots()
axis.plot(d_history, label="Discriminator")
axis.plot(g_history, label="Generator")
axis.set(xlabel="Training step", ylabel="Binary cross-entropy")
axis.legend()
fig.tight_layout()
fig.savefig("gan_losses.png")
plt.close(fig)
```

| Output | Inspection |
| --- | --- |
| `samples_epoch_1.png`, `samples_epoch_2.png` | Comparison of image shapes and sample diversity across epochs using the same 16 noise vectors |
| `gan_losses.png` | Comparison of generator and discriminator losses by training step, alongside inspection of the generated images |

---

## 4. Training Challenges and GAN Variants

- **Loss oscillation** — Fluctuation in losses as the generator and discriminator change in response to each other.
- **Mode collapse** — Repetition of similar outputs across different noise vectors, with reduced coverage of patterns in the training data.

| Model | Training objective | Constraint |
| --- | --- | --- |
| Standard GAN | Binary cross-entropy for real/fake discrimination and generator training | Binary classification targets |
| WGAN | Wasserstein-based objective using real-valued critic scores | Weight clipping to impose a Lipschitz constraint on the critic |
| WGAN-GP | WGAN objective with a gradient penalty on the critic | Penalty when the critic's input-gradient norm departs from `1` at interpolations between real and generated samples |

---

## 5. MNIST Extension and Applications

Training of a fresh GAN on the first 10,000 MNIST training images using the same architecture and normalization. A separate working directory for keeping the camel and digit output files.

```bash
python train_gan.py --mnist
```

| Application | Example |
| --- | --- |
| Image generation | Camel sketches, handwritten digits, and synthetic faces |
| Image-to-image translation | pix2pix and CycleGAN |
| Single-image super-resolution | SRGAN |

</details>

<details>
<summary>KOR (한국어 버전)</summary>

# Week 06 — 생성적 적대 신경망과 이미지 생성

TensorFlow와 Keras를 활용한 합성곱 기반 생성적 적대 신경망(GAN) 구현. Quick, Draw! 낙타 스케치와 MNIST 손글씨 숫자 생성 및 생성 이미지·학습 손실 확인.

## 1. GAN 구조

두 신경망의 경쟁적 학습을 통한 데이터 분포 학습 및 새 샘플 생성.

| 구성 요소 | 입력 | 역할 |
| --- | --- | --- |
| 생성자(Generator, G) | 표준정규분포에서 추출한 100차원 벡터 `z` | `28 × 28 × 1` 흑백 이미지 생성 |
| 판별자(Discriminator, D) | 실제 이미지 또는 G가 생성한 이미지 | 실제 이미지와 생성 이미지를 구분하는 이진 분류 로짓 출력 |

```text
랜덤 노이즈 z → 생성자 → 생성 이미지 ─┐
                                     ├→ 판별자 → 실제/생성 판별 점수
학습 데이터 ──────────→ 실제 이미지 ──┘
```

---

## 2. 생성자·판별자의 교대 학습

| 학습 단계 | 목표 라벨 | 갱신 대상 |
| --- | --- | --- |
| 판별자 학습 | 실제 이미지 → `1`, 생성 이미지 → `0` | 판별자 가중치 |
| 생성자 학습 | 판별자에 입력한 생성 이미지 → 목표값 `1` | 생성자 가중치 |

두 단계에서 이진 교차 엔트로피 사용. 생성자 학습 단계에서는 판별자 가중치를 고정한 상태로, 판별자 출력을 거쳐 생성자까지 역전파.

---

## 3. Quick, Draw! 낙타 스케치 생성

### 3.1. 데이터와 모델 설정

| 항목 | 설정 |
| --- | --- |
| 학습 데이터 | Google Quick, Draw! `camel` 범주의 처음 10,000개 스케치 |
| 입력 전처리 | `(N, 28, 28, 1)` 형태로 변환하고 픽셀값을 `[0, 255]`에서 `[-1, 1]`로 정규화 |
| 생성자 | `7 × 7 × 64` 특징 맵으로 변환한 뒤 전치 합성곱 2개와 `tanh` 출력 적용 |
| 판별자 | 스트라이드 합성곱, Leaky ReLU, 드롭아웃, 단일 로짓 출력 적용 |
| 옵티마이저 | RMSprop 사용. 판별자 학습률 `0.0008`, 생성자 학습률 `0.0004` |
| 학습 설정 | 기본 동작 확인용 배치 크기 `64`, `2` 에폭 |

### 3.2. 환경 준비와 실행

라이브러리 설치 및 공식 NumPy 비트맵 파일을 코드에서 사용하는 이름으로 다운로드.

```bash
python -m pip install "tensorflow>=2.16" numpy matplotlib
curl -fL https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/camel.npy -o full_numpy_bitmap_camel.npy
```

아래 코드를 데이터 파일과 같은 폴더에 `train_gan.py`로 저장한 뒤 실행.

```bash
python train_gan.py
```

### 3.3. 학습 및 시각화 코드

```python
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = ArgumentParser()
parser.add_argument("--mnist", action="store_true")
args = parser.parse_args()
keras.utils.set_random_seed(42)

if args.mnist:
    (raw, _), _ = keras.datasets.mnist.load_data()
else:
    raw = np.load("full_numpy_bitmap_camel.npy", mmap_mode="r")

images = np.asarray(raw[:10_000], dtype=np.float32).reshape(-1, 28, 28, 1)
images = images / 127.5 - 1.0
batches = tf.data.Dataset.from_tensor_slices(images).shuffle(len(images)).batch(64)

generator = keras.Sequential([
    keras.Input(shape=(100,)),
    layers.Dense(7 * 7 * 64, activation="relu"),
    layers.Reshape((7, 7, 64)),
    layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu"),
    layers.Conv2DTranspose(1, 4, strides=2, padding="same", activation="tanh"),
])
discriminator = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(64, 5, strides=2, padding="same"),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Conv2D(128, 5, strides=2, padding="same"),
    layers.LeakyReLU(negative_slope=0.2),
    layers.Flatten(),
    layers.Dropout(0.4),
    layers.Dense(1),
])

bce = keras.losses.BinaryCrossentropy(from_logits=True)
d_optimizer = keras.optimizers.RMSprop(learning_rate=0.0008)
g_optimizer = keras.optimizers.RMSprop(learning_rate=0.0004)
d_history, g_history = [], []
fixed_noise = tf.random.normal((16, 100))

for epoch in range(1, 3):
    for real in batches:
        noise_shape = (tf.shape(real)[0], 100)
        fake = generator(tf.random.normal(noise_shape), training=False)

        with tf.GradientTape() as tape:
            real_scores = discriminator(real, training=True)
            fake_scores = discriminator(fake, training=True)
            d_loss = (
                bce(tf.ones_like(real_scores), real_scores)
                + bce(tf.zeros_like(fake_scores), fake_scores)
            ) / 2
        d_weights = discriminator.trainable_weights
        d_optimizer.apply_gradients(zip(tape.gradient(d_loss, d_weights), d_weights))

        with tf.GradientTape() as tape:
            fake = generator(tf.random.normal(noise_shape), training=True)
            fake_scores = discriminator(fake, training=False)
            g_loss = bce(tf.ones_like(fake_scores), fake_scores)
        g_weights = generator.trainable_weights
        g_optimizer.apply_gradients(zip(tape.gradient(g_loss, g_weights), g_weights))

        d_history.append(float(d_loss))
        g_history.append(float(g_loss))

    samples = (generator(fixed_noise, training=False).numpy() + 1.0) / 2.0
    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for axis, sample in zip(axes.flat, samples):
        axis.imshow(sample[..., 0], cmap="gray", vmin=0, vmax=1)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(f"samples_epoch_{epoch}.png")
    plt.close(fig)

fig, axis = plt.subplots()
axis.plot(d_history, label="Discriminator")
axis.plot(g_history, label="Generator")
axis.set(xlabel="Training step", ylabel="Binary cross-entropy")
axis.legend()
fig.tight_layout()
fig.savefig("gan_losses.png")
plt.close(fig)
```

| 출력 파일 | 확인 항목 |
| --- | --- |
| `samples_epoch_1.png`, `samples_epoch_2.png` | 동일한 노이즈 벡터 16개를 사용한 에폭별 이미지 형태와 샘플 다양성 비교 |
| `gan_losses.png` | 학습 단계별 생성자·판별자 손실 추이 비교 및 생성 이미지와 함께 해석 |

---

## 4. 학습의 어려움과 GAN 변형

- **손실 진동(Loss Oscillation)** — 생성자와 판별자가 서로의 변화에 대응하면서 손실값이 반복적으로 변동하는 현상.
- **모드 붕괴(Mode Collapse)** — 서로 다른 노이즈를 넣어도 비슷한 결과가 반복되어 학습 데이터의 다양한 패턴을 충분히 반영하지 못하는 현상.

| 모델 | 학습 목적 | 제약 방식 |
| --- | --- | --- |
| 기본 GAN | 실제·생성 이미지 판별과 생성자 학습에 이진 교차 엔트로피 사용 | 이진 분류 목표 라벨 사용 |
| WGAN | 평가자(Critic)의 실수 점수를 활용한 Wasserstein 기반 목적 함수 | 평가자의 Lipschitz 제약을 위한 가중치 클리핑 |
| WGAN-GP | WGAN 목적 함수에 평가자의 기울기 페널티 추가 | 실제·생성 이미지 사이의 보간 지점에서 입력 기울기 노름이 `1`에서 벗어나는 정도에 페널티 적용 |

---

## 5. MNIST 확장 실습과 응용

같은 모델 구조와 정규화를 사용해 MNIST 학습 이미지 처음 10,000개로 새 GAN 학습. 낙타와 숫자 생성 결과를 따로 보관할 경우 별도 작업 폴더에서 실행.

```bash
python train_gan.py --mnist
```

| 응용 분야 | 예시 |
| --- | --- |
| 이미지 생성 | 낙타 스케치, 손글씨 숫자, 합성 얼굴 생성 |
| 이미지 변환 | pix2pix, CycleGAN |
| 단일 이미지 초해상도 복원 | SRGAN |

</details>
