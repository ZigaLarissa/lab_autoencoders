# Network Anomaly Detection using Autoencoders

This project implements an unsupervised machine learning approach to detect anomalies in network data using autoencoders. Autoencoders are utilized to reconstruct normal network patterns and identify anomalies by measuring reconstruction errors. The dataset used is the **KDD99** dataset, which contains labeled network traffic data. While labels were used for evaluation, the training process remained unsupervised.

## Features

- **Unsupervised Learning**: Detect anomalies without the need for labeled data, addressing real-world challenges like zero-day attack detection.
- **Deep Autoencoder**: Leverage encoder-decoder architecture with dropout layers for effective anomaly detection.
- **TensorBoard Visualization**: Monitor training progress and network performance through TensorBoard.

## Dataset

The **KDD99** dataset is used to simulate realistic anomaly detection problems. Anomalies were reduced to a small percentage to mimic real-world scenarios. Data preprocessing includes:
- One-hot encoding of categorical variables.
- Scaling input features using MinMaxScaler to the range [0, 1].

## Model Architecture

The autoencoder consists of:
1. **Encoder Network**: Reduces input dimensions to a bottleneck latent space.
2. **Latent Space**: Captures essential data characteristics.
3. **Decoder Network**: Reconstructs input data from the latent space.

Dropout layers are included to prevent overfitting, and symmetry is maintained between the encoder and decoder.

### Hyperparameters

- **Batch Size**: 512
- **Latent Dimension**: 4
- **Epochs**: 10
- **Learning Rate**: 0.00001 (using Adam optimizer)

## Workflow

1. **Data Preprocessing**: Prepare the data by encoding, normalizing, and splitting into training and testing sets.
2. **Model Training**:
   - Train the autoencoder on normal data using reconstruction loss (MSE).
   - Validate using unseen data and visualize training loss convergence.
3. **Anomaly Detection**:
   - Reconstruct test data and compute reconstruction scores (MSE).
   - Set thresholds for labeling test points as anomalies.
4. **Evaluation**:
   - Visualize reconstruction error distributions.
   - Compare results with ground truth labels to evaluate performance.

## Results

- The autoencoder successfully identified anomalies in the dataset by leveraging reconstruction scores.
- Key metrics such as the ROC curve and confusion matrix were used to assess model performance.

## Visualization

TensorBoard logs were used for:
- Monitoring training and validation loss.
- Visualizing model architecture and layer summaries.

Example plots:
- Training vs. Validation Loss
- Reconstruction Score Distribution


## Dependencies

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- TensorBoard

## Future Improvements

- Experiment with different latent dimensions for enhanced feature extraction.
- Implement other anomaly detection algorithms for comparison.
- Extend to other datasets for broader applicability.

## Acknowledgements

This lab was guided by:
- **Ananth Sankar**, Solutions Architect at NVIDIA
- **Eric Harper**, Solutions Architect, Global Telecoms at NVIDIA
