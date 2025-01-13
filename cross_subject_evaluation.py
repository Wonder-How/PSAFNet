import numpy as np
import torch
import os
from torchinfo import summary
from data_processing import create_cross_subject_dataloaders
from train import train, test
from utils import set_seed
from PSAFNet import PSAFNet
from my_config import config

# Ensure CUDA launch blocking for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set random seed for reproducibility
set_seed(config.seed)

# Data type and file paths for cross-subject EEG data
data_type = "unica"  # Processed data type: 'unica'
file_paths = [
    fr"D:\machine learning\EEG_video\{data_type}_data\cw_1.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\ghr_2.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\gr_3.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\kx_4.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\pbl_5.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\sjt_6.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\wxc_7.mat",
    fr"D:\machine learning\EEG_video\{data_type}_data\xxb_8.mat"
]

# Define device (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the PSAFNet model
model = PSAFNet(
    stage_timepoints=config.stage_timepoints,
    lead=config.num_channels,
    time=config.num_timepoints
).to(device)

# Display model summary
summary(model)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Initialize variables to store results
fold_results = []
k_folds = 8  # Number of folds (subjects)
fold_accuracies = []
fold_hit_rates = []
fold_false_alarm_rates = []

# Loop through each fold (subject)
for i in range(k_folds):
    print(f"Testing on Subject {i + 1}")

    # Create cross-subject data loaders
    train_loader, val_loader, test_loader = create_cross_subject_dataloaders(
        file_paths,
        window_length=(config.num_timepoints / config.fs),
        batch_size=config.batchsize,
        test_index=i
    )

    # Train the model
    train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=config.num_epochs)

    # Test the model
    test_accuracy, hit_rate, false_alarm_rate = test(model, device, test_loader)

    # Convert rates to percentages and store results
    test_accuracy *= 100
    fold_accuracies.append(test_accuracy)
    fold_hit_rates.append(hit_rate * 100)
    fold_false_alarm_rates.append(false_alarm_rate * 100)

    # Print results for the current subject
    print(
        f'Subject {i + 1}: Accuracy = {test_accuracy:.6f}%, Hit Rate = {hit_rate * 100:.6f}%, False Alarm Rate = {false_alarm_rate * 100:.6f}%')

# Print individual subject results
print('\nIndividual Subject Results:')
for i in range(k_folds):
    print(
        f'Subject {i + 1}: Accuracy = {fold_accuracies[i]:.6f}%, Hit Rate = {fold_hit_rates[i]:.6f}%, False Alarm Rate = {fold_false_alarm_rates[i]:.6f}%')

# Calculate overall mean and standard deviation for metrics
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)
mean_hit_rate = np.mean(fold_hit_rates)
std_hit_rate = np.std(fold_hit_rates)
mean_false_alarm_rate = np.mean(fold_false_alarm_rates)
std_false_alarm_rate = np.std(fold_false_alarm_rates)

# Print overall results
print('\nOverall Results (Mean ± Std):')
print(f'Accuracy: {mean_accuracy:.6f}% ± {std_accuracy:.6f}%')
print(f'Hit Rate: {mean_hit_rate:.6f}% ± {std_hit_rate:.6f}%')
print(f'False Alarm Rate: {mean_false_alarm_rate:.6f}% ± {std_false_alarm_rate:.6f}%')