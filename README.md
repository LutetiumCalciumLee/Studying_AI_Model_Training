<details>
<summary>ENG (English Version)</summary>

## **Variational Autoencoder (VAE)**

### **What is VAE?**
- A **generative model** that learns a **probabilistic latent space** for data generation.
- Unlike regular autoencoders, VAE learns **distributions** (mean & variance) instead of fixed latent vectors.
- Enables smooth sampling and interpolation in latent space for generating realistic new data (e.g., faces).

### **Core Architecture**
- **Encoder**: Outputs `z_mean` and `z_log_var` representing a latent Gaussian distribution.
- **Sampling Layer**: Applies the reparameterization trick:  
  `z = z_mean + epsilon * exp(z_log_var * 0.5)`
- **Decoder**: Reconstructs input data from sampled latent vector `z`.
- **Loss Function**:
  - **Reconstruction Loss**: Measures how well output matches input.
  - **KL Divergence**: Encourages latent space to follow standard normal distribution (N(0,1)).

### **TensorFlow/Keras Implementation**
- **Structure**: Custom `VAE` class with encoder, sampling layer, and decoder.
- **Training**:
  - Compile with Adam optimizer.
  - Custom `train_step()` to compute both losses.
- **Generation**:
  - After training, use random latent vectors to generate new outputs from the decoder.

```python
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)

random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
generated_images = vae.decoder(random_latent_vectors)
````

### **Application: Face Generation**

* Trained on celebrity face dataset.
* **Model**: Deep CNN with `Conv2D`, `Conv2DTranspose`, `BatchNorm`, `LeakyReLU`.
* **Latent Space**: 200-dimensional.
* **Visualization**: Uses custom callback (`ImageGenerator`) to generate face grids after each epoch.
* **Outcome**: Produces realistic face images by decoding latent samples.

</details>

<details>
<summary>KOR (한국어 버전)</summary>

## **변분 오토인코더 (VAE)**

### **VAE란?**

* 데이터를 **확률적 잠재 공간(latent space)**에 인코딩하고 다시 복원하는 **생성 모델**.
* 일반 오토인코더와 달리, VAE는 잠재 벡터 대신 **분포의 평균과 분산**을 학습한다.
* 잠재 공간에서 샘플링을 통해 **새로운 데이터를 생성**할 수 있어 다양성과 연속성이 확보됨.

### **핵심 구조**

* **인코더 (Encoder)**: 입력 데이터를 `z_mean`과 `z_log_var`라는 두 벡터로 변환.
* **샘플링 레이어**: 아래 수식으로 랜덤 벡터 `z` 생성
  `z = z_mean + ε * exp(0.5 * z_log_var)`
* **디코더 (Decoder)**: 잠재 벡터 `z`를 원래 형태로 복원.
* **손실 함수**:

  * **재구성 손실**: 출력이 입력을 얼마나 잘 복원하는지 측정.
  * **KL 발산 (Kullback-Leibler Divergence)**: 잠재 공간이 표준 정규분포(N(0,1))를 따르도록 유도.

### **TensorFlow/Keras 구현**

* **구성**: 인코더, 샘플링, 디코더를 포함한 커스텀 `VAE` 클래스 사용.
* **학습**:

  * Adam 옵티마이저로 컴파일.
  * 사용자 정의 `train_step()`으로 손실 계산.
* **이미지 생성**:

  * 훈련 후, 무작위 잠재 벡터를 디코더에 넣어 새로운 이미지 생성.

```python
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=30, batch_size=128)

random_latent_vectors = tf.random.normal(shape=(10, latent_dim))
generated_images = vae.decoder(random_latent_vectors)
```

### **응용 예시: 얼굴 생성**

* **학습 데이터**: 유명인 얼굴 이미지 데이터셋.
* **모델 구성**: CNN 기반 VAE, `Conv2D`, `Conv2DTranspose`, `BatchNormalization`, `LeakyReLU` 사용.
* **잠재 공간 차원**: 200차원 (`Z_DIM = 200`)
* **시각화 방법**:

  * 에폭마다 샘플 이미지를 저장하는 `ImageGenerator` 콜백 사용.
  * 학습이 진행될수록 점점 더 사실적인 얼굴 이미지 생성 가능.

</details>
