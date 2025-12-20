<details>
<summary>ENG (English Version)</summary>

## **Generative Adversarial Network (GAN)**

### **What is GAN?**
- GAN consists of two neural networks: a **Generator** and a **Discriminator**.
- The Generator creates fake data from noise; the Discriminator tries to distinguish real from fake.
- As training progresses, the Generator gets better at fooling the Discriminator.

### **Core Components**
- **Generator (G)**: Converts a latent noise vector (`z`) into a realistic data sample (e.g., 28x28 image).
- **Discriminator (D)**: Acts as a binary classifier that outputs the probability of a sample being real.

### **Training Process**
1. Train Discriminator with **real images** → label as `1`.
2. Train Discriminator with **fake images** → label as `0`.
3. Train Generator to **fool the Discriminator** → aim to produce images classified as real (`1`).

### **Python Implementation**
- **Goal**: Train a GAN to generate 28x28 camel drawings from the “Quick, Draw!” dataset.
- **Model**:
  - Uses `Conv2D`, `Conv2DTranspose`, and custom hyperparameters.
  - Encapsulated in a custom `GAN` class with build/train methods.
- **Training**:
  - Optimizers like `RMSprop` used.
  - Losses for both Generator and Discriminator are monitored and visualized.

```python
gan = GAN(
    input_dim = (28,28,1),
    discriminator_learning_rate = 0.0008,
    generator_learning_rate = 0.0004,
    z_dim = 100
)
gan.train(images, batch_size=64, epochs=2, run_folder=RUN_FOLDER)

# Visualize loss
plt.plot([x[0] for x in gan.d_losses], color='black')
plt.plot([x[0] for x in gan.g_losses], color='orange')
plt.show()
````

### **Challenges**

* **Oscillating Loss**: Loss values may fluctuate without convergence.
* **Mode Collapse**: Generator outputs repetitive samples with low diversity.

### **Advanced Variants**

* **WGAN**: Uses Wasserstein distance and weight clipping for stable learning.
* **WGAN-GP**: Improves WGAN by using gradient penalty and removing batch normalization in the critic.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## **생성적 적대 신경망 (GAN)**

### **GAN이란?**

* **생성자(Generator)** 와 **판별자(Discriminator)** 라는 두 개의 신경망이 서로 경쟁하며 학습하는 **생성 모델**.
* 생성자는 랜덤 노이즈로부터 가짜 데이터를 생성하고, 판별자는 그것이 진짜인지 가짜인지 구별한다.
* 반복 훈련을 통해 생성자는 점점 더 진짜같은 데이터를 생성하게 된다.

### **핵심 구성 요소**

* **생성자 (G)**: 잠재 벡터(`z`)를 받아서 진짜 같은 이미지를 생성.
* **판별자 (D)**: 입력이 진짜인지 가짜인지 판단하는 이진 분류기.

  * 진짜 → `1`, 가짜 → `0`

### **학습 과정**

1. **진짜 이미지**로 판별자 학습 (`label = 1`)
2. **생성된 이미지**로 판별자 학습 (`label = 0`)
3. 판별자는 고정하고 **생성자만 업데이트** → 생성자가 판별자를 속이도록 학습 (`label → 1`)

### **파이썬 구현 개요**

* **목표**: "Quick, Draw!"의 낙타 이미지(28x28)를 생성하는 GAN 모델 학습.
* **모델 구성**:

  * `Conv2D`, `Conv2DTranspose` 사용.
  * `GAN` 클래스에 생성자, 판별자, 훈련 루틴 포함.
* **학습 및 시각화**:

  * `rmsprop` 옵티마이저 사용.
  * Generator/Discriminator의 손실 추이를 시각화.

```python
gan = GAN(
    input_dim = (28,28,1),
    discriminator_learning_rate = 0.0008,
    generator_learning_rate = 0.0004,
    z_dim = 100
)
gan.train(images, batch_size=64, epochs=2, run_folder=RUN_FOLDER)

# 손실 시각화
plt.plot([x[0] for x in gan.d_losses], color='black')
plt.plot([x[0] for x in gan.g_losses], color='orange')
plt.show()
```

### **학습의 어려움**

* **Loss 진동(Oscillation)**: 손실값이 수렴하지 않고 계속 변동.
* **모드 붕괴(Mode Collapse)**: 생성자가 다양성 없는 유사한 결과만 반복 생성.

### **고급 GAN 모델**

* **WGAN**: Wasserstein 거리와 가중치 클리핑으로 안정성 향상.
* **WGAN-GP**: Gradient Penalty 기법 도입, 판별자에서 BatchNorm 제거.

</details>
