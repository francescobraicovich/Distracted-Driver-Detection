import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import random
from sklearn.metrics import precision_recall_fscore_support
import time
from torchvision.io import read_image
from IPython.display import clear_output
from tqdm import tqdm



def train(model, train_dataset, val_dataset, num_epochs, batch_size, workers):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizing all parameters
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    metrics = pd.DataFrame(columns=['Epoch', 'Training Loss', 'Test Loss', 
                                'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    

    # train loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    start_time = time.time()

    metrics = pd.DataFrame(columns=['Epoch', 'Training Loss', 'Test Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs_cnn, labels) in enumerate(tqdm(train_loader, desc="Training Batches")):
            optimizer.zero_grad()
            outputs = model(inputs_cnn)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.6f}')

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.6f}')

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            val_loss = 0
            all_labels = []
            all_preds = []
            for inputs_cnn, labels in val_loader:
                outputs = model(inputs_cnn)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

            metrics.loc[len(metrics)] = {'Epoch': epoch+1, 'Training Loss': avg_loss, 'Test Loss': avg_val_loss, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1_score}
            
            # Clear the previous output in the notebook
            clear_output(wait=True)

            # Create a figure with 3 axes: 2 for the plots, 1 for the table below the first plot
            fig, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})

            # Plot training and test loss on the top-left axis
            metrics.plot(x='Epoch', y=['Training Loss', 'Test Loss'], ax=axs[0, 0])
            axs[0, 0].set_title("Training and Test Loss")

            # Plot accuracy, precision, recall, and F1-score on the top-right axis
            metrics.plot(x='Epoch', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'], ax=axs[0, 1])
            axs[0, 1].set_title("Classification Metrics on Validation Set")

            # Remove legends from the plots for a cleaner look
            axs[0, 0].legend(loc='upper right')
            axs[0, 1].legend(loc='upper left')

            # Hide the axis below the left plot (used for the table)
            axs[1, 0].axis("off")

            # Convert the last row of metrics to a table format, rounded to three decimals
            last_metrics = metrics.iloc[[-1]].round(3).T  # Select the last row, round, and transpose for display
            last_metrics.columns = ['Last Epoch Metrics']  # Rename column for clarity
            last_metrics = last_metrics.reset_index()  # Reset index to remove the index label

            # Add the table to the bottom-left axis
            table = axs[1, 0].table(cellText=last_metrics.values,
                                    colLabels=last_metrics.columns,
                                    cellLoc='center',
                                    loc='center')

            # Customize the table for better readability
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)  # Adjusts the width and height of the cells

            # Hide the empty bottom-right plot area for a cleaner look
            axs[1, 1].axis("off")

            # Adjust layout
            plt.subplots_adjust(hspace=0.3, bottom=0.25)
            plt.show()
            plt.close(fig)  # Close figure to free up memory

    end_time = time.time()
    print(f"Training completed in: {end_time - start_time:.2f} seconds")

    return model, metrics
