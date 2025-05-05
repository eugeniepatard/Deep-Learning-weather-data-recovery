# Deep Learning for Weather Data Reconstruction

This project was developed as part of the **Deep Learning module** in the MSc *Applied Computational Science and Engineering* at **Imperial College London**. It was completed under a **24-hour assessment constraint**, simulating real-world time-limited project execution.

## 📘 Overview

Large-scale weather datasets often suffer from missing entries due to sensor failures or data transmission issues. This project explores deep learning-based methods for **temporal gap-filling**—reconstructing missing values in time series weather data. Using PyTorch, the model is trained to learn from corrupted and uncorrupted historical data to predict and restore the gaps in unseen sequences.

## 🧠 Problem Statement

Given daily weather data for a European city over a period of ~40 years, a portion of the data has been corrupted with missing entries. The training data includes three decades of corrupted datasets along with their original, uncorrupted versions. The final test set contains a fourth corrupted decade, for which the original values are no longer available.

The objective is to build a neural network that can infer and recover missing values in this test dataset, leveraging patterns learned from the historical data.

## 🗂️ Project Structure

- `Weather-data-reconstruction-dl.ipynb`: Contains all analysis, visualizations, model design, training loop, and final predictions.
- `test_set_nogaps.csv`: Output file with the reconstructed weather values for the test set.
- `References.md`: A record of all external tools and sources used during the project.
- `training_set/`: Folder containing input training data (`_nogaps.csv` and corrupted `.csv` files).
- `test_set.csv`: The corrupted test dataset to be reconstructed.

## 📊 Data Features

Each file includes daily records of:
- `date`
- `cloud_cover`
- `sunshine`
- `global_radiation`
- `max_temp`
- `mean_temp`
- `min_temp`
- `precipitation`
- `pressure`

## ⚙️ Methodology

1. **Exploratory Analysis**  
   - Loaded three decades of weather data, both with and without missing values.  
   - Plotted each variable over time and compared corrupted vs. uncorrupted versions to visualize the effect of missing data.  
   - Generated histograms to compare distributions across corrupted and clean data.  
   - Computed and visualized a correlation matrix across sequences of days to inform sequence length (ultimately choosing `seq_len = 4` for strong temporal correlation).

2. **Data Preparation**  
   - Concatenated all decades into a single dataset and applied `RobustScaler` for normalization.  
   - Padded the data at the start and end to allow sequence slicing without data loss due to boundaries.  
   - Constructed a custom `Dataset` class to handle interpolation, masking of missing values (`mask == 1` for missing), and supervised pairing with clean labels.  
   - Split the dataset into training and validation sets (90/10 split) and used PyTorch `DataLoader`s with batch size 64.

3. **Model Design**  
   - Implemented an LSTM-based recurrent neural network (`WeatherRNN`) with two stacked LSTM layers and a dropout of 0.6.  
   - The model receives sequences of length 4 and predicts the same length, masking known values to focus only on imputation.  
   - Defined a custom masked MSE loss function to compute reconstruction error only where data was originally missing.  
   - Trained using Adam optimizer with learning rate `1e-4`, weight decay `1e-5`, and mean squared error loss over 60 epochs.

4. **Evaluation & Output**  
   - Evaluated model predictions on a held-out test set, visualized reconstructed time series, and compared them to input corrupted sequences.  
   - Demonstrated ability to recover missing values accurately across multiple weather variables.  
   - Final outputs were stored with preserved shape and format, suitable for saving in `test_set_nogaps.csv`.

## 📊 Results

- Initially explored a **Feedforward Neural Network (FFN)** but found it insufficient due to lack of temporal modeling.  
  → Switched to a **Recurrent Neural Network (RNN)** using LSTM cells to capture time dependencies between days.

- **Normalization** was crucial:  
  → Without normalization, the model produced high loss and poor predictions.  
  → Applied `RobustScaler` to both training and test sets for improved convergence.

- Constructed training sequences using `seq_len = 4` based on correlation analysis:  
  → Correlation matrix indicated strongest relationships across 4-day windows.  
  → Concatenated all decades for sequence construction, which occasionally spanned across decades.  
  → This introduced minor edge inconsistencies, but was negligible due to short `seq_len`.

- Discovered the importance of **padding**:  
  → Without padding, model failed to reconstruct some rows at the beginning/end of sequences.  
  → Applied padding before slicing sequences to maintain full temporal coverage.

- **Hyperparameter tuning**:  
  → Started with short training runs to observe loss curves.  
  → Used a validation split to detect overfitting.  
  → Overfitting was partially mitigated by increasing dropout to 0.6.  
  → Final model settings:  
    - Learning rate: `1e-4`  
    - Weight decay: `1e-5`  
    - Dropout: `0.6`

- Despite some overfitting, the final model provided **consistent and coherent reconstructions** of missing weather data.  
  → Predictions matched the trends and distributions of the clean sequences closely.  
  → With more time, further tuning and architectural experiments (e.g., Transformer models or decade-specific training) could improve generalization.


## 📚 Skills & Tools

- Python, PyTorch
- Time Series Analysis
- Data Imputation
- Neural Network Architecture Design
- Data Visualization (Matplotlib, Seaborn)
- Git, GitHub

## 🏫 Academic Context

This project was submitted as part of a **24-hour individual coursework** for the **Deep Learning** module in the **MSc in Applied Computational Science and Engineering** at **Imperial College London**. It reflects a practical, time-constrained application of machine learning techniques to solve real-world data challenges.
