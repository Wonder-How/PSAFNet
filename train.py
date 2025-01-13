import time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm

def train(model, device, train_loader, val_loader, criterion, optimizer, num_epochs=100, save_path='best_model.pth', show=True):
    best_val_loss = float('inf')
    with tqdm(total=num_epochs, desc='Training Progress', unit='epoch') as pbar:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            alpha = 1.5
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = inputs.unsqueeze(1)
                optimizer.zero_grad()
                outputs, similarity_loss = model(inputs)
                CE_loss = criterion(outputs, labels)
                loss = CE_loss + alpha * similarity_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            val_loss, val_accuracy = validate(model, device, val_loader, criterion)
            if show:
                pbar.set_postfix({
                    'Train Loss': running_loss / len(train_loader),
                    'Val Loss': val_loss,
                    'Val Accuracy': val_accuracy
                })
            pbar.update(1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f'Saving model with val_loss {val_loss:.4f} and val_accuracy {val_accuracy} at epoch {epoch + 1}')

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.unsqueeze(1)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    val_accuracy = accuracy_score(all_labels, all_preds)
    return val_loss / len(val_loader), val_accuracy

def test(model, device, test_loader, load_path='best_model.pth'):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    all_preds = []
    all_labels = []
    prediction_times = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.unsqueeze(1)
            start_time = time.time()
            outputs, _ = model(inputs)
            end_time = time.time()
            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    hit_rate = tp / (tp + fn) if (tp + fn) != 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) != 0 else 0
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Hit Rate (TPR): {hit_rate:.4f}')
    print(f'False Alarm Rate (FPR): {false_alarm_rate:.4f}')
    print(f'Average Prediction Time per Sample: {np.mean(prediction_times):.6f} seconds')
    return accuracy, hit_rate, false_alarm_rate
