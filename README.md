# Accent-Classification-and-Conversion

**Authors**: Arvind Raghavendran (MDS202214), Swastika Mohapatra (MDS202245)

---

## Introduction to Audio Data

Audio data is a versatile medium utilized across various domains, including speech recognition, voice translations, and music analysis. However, working with audio data presents several challenges, such as raw data, device/environmental/random noise additions, and variable durations.

## Mel-Filter Cepstrum Coefficients (MFCC)

MFCC is based on the human auditory system's response to audio and is effective for capturing spectral features, such as the shape of the vocal tract. It offers advantages like dimensionality reduction, robustness to noise, and computational efficiency. However, it cannot capture finer temporal differences like pitch effectively.

## Spectrograms

Spectrograms provide a visual representation of the frequency content of an audio signal over time, capturing both temporal and spectral information. While they are suitable for tasks requiring detailed analysis of audio features, such as accent classification and music genre classification, they can be computationally expensive and require careful hyperparameter tuning.

## Accent Classification via MFCCs (Task 1)

### Data Description

The dataset consists of audio recordings of speakers from different countries with diverse accents. Our focus is on Indian and British accents, comprising 742 recordings speaking the same passage. We extracted 13 MFCCs across windowed timestamps to generate feature matrices and compressed information by taking row-wise averages to obtain feature vectors.

### Model Constructions

We experimented with simple linear models and 1-D Convolutional layers to capture contextual information. Batch Normalization and Dropout were applied to ensure steady training and prevent overfitting. The model was compiled with Cross Entropy Loss and Adam optimizer.

### Training and Evaluation

The best model consisted of 4 convolutional layers followed by Global Average Pooling and then 3 Dense layers. It was trained in batches of 32 for 1000 epochs. Out-of-sample metrics showed Precision: 91%, Recall: 86%, and Accuracy: 88%. However, the model struggled to capture finer nuances in accents.

## Accent Classification via Spectrograms (Task 2)

### Issues with Using MFCCs

MFCCs capture high-level vocal tract information but may not capture subtle pitch and frequency changes crucial for accent classification. Spectrograms, on the other hand, represent audio signals in the time-frequency domain, capturing both temporal and spectral features simultaneously.

### Data Construction

We experimented with various hyperparameters and applied the Hanning algorithm to obtain windows for applying Short-Time Fourier Transform (STFT). The resulting matrices were padded with zeros to have inputs of the same size.

### Training and Evaluation

We used the same model architecture as in Task 1 but replaced the 1D layers with 2D layers. MaxPooling was introduced at each layer to reduce the size of feature maps. Out-of-sample metrics showed significant improvement, with Precision: 97.68%, Recall: 98.13%, and Accuracy: 97.97%.

## Accent Conversion (Task 3)

### Model Considerations

Accent conversion involves compressing input audio into a latent space and reconstructing it back to the target domain. Autoencoder frameworks or CNN+LSTMs were considered suitable for this task.

### Data Description

We extracted 26 MFCCs for higher representation and padded the matrices with zeros. The data were normalized to reduce overhead, and mean and variances were stored.

### Model Constructions

The core training ideas remained the same, with additional normalization of inputs for smoother training. We experimented with different loss functions and model architectures, ultimately choosing a model with an encoder-decoder architecture and a latent embedding.

### Training and Evaluation

The model showed improvement in learning, but reconstruction showed smoothening of features. We replaced the vanilla loss functions with an adversarial loss and introduced skip connections to improve model performance.

## Further Improvements

While significant improvements were achieved, further enhancements are possible by exploring additional datasets, deeper architectures, and higher sampling rates. Transfer learning on audio data could also be explored for faster and powerful latent representations.
