import matplotlib as plt
import torch
from data import ECGDataSet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def plot_residuals(predictions, actuals):
    # Check if inputs are tensors, convert only if necessary
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(actuals, torch.Tensor):
        actuals = actuals.cpu().numpy()

    residuals = predictions - actuals
    
    # Ensure inputs are NumPy arrays
    if(not isinstance(predictions, np.ndarray) or 
       not isinstance(actuals, np.ndarray) or 
       not isinstance(residuals, np.ndarray)):
        raise ValueError("Inputs must be either PyTorch tensors or NumPy arrays.")
    
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    plt.scatter(actuals, residuals)
    plt.title(f'MSE {mse:.2e}, $R^2$ = {r2:.2f}', fontsize=12)
    plt.xlabel('Actual Value')
    plt.ylabel('Residuals')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def plot_signal_pred(model, lead_num, dataset: ECGDataSet):
    lead_start = None
    lead_end = None
    
    if(lead_num != None):
        lead_start = (lead_num - 1) * 5000
        lead_end = lead_num * 5000
    
    # make a precition with the model and visually compare to the validation set
    sample_id = np.random.randint(low = 0, high = len(dataset), dtype=int)
    X_val,y_val = dataset[sample_id]

    # Set figure size and create two stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Time and plot actual signal
    time = np.linspace(0, 10, 5000)
    axes[0].plot(time, y_val[lead_start:lead_end], color='blue', label="Actual")
    axes[0].set_title("Actual Signal")  # Title for the first plot
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Signal (mV)")
    axes[0].legend()

    # Prediction
    with torch.no_grad():
        y_pred = model(X_val.unsqueeze(0))  # Unsqueeze to add "batch"
    print(y_pred.shape)
    axes[1].plot(time, y_pred.squeeze(0)[lead_start:lead_end], color='red', label="Prediction")
    axes[1].set_title("Prediction Signal")  # Title for the second plot
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Signal (mV)")
    axes[1].legend()

    # difference
    print(y_pred.shape)
    axes[2].plot(time, y_pred.squeeze(0)[lead_start:lead_end] - y_val[lead_start:lead_end], color='red', label="Prediction")
    axes[2].set_title("Error")  # Title for the second plot
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Signal (mV)")
    axes[2].legend()

    # Adjust layout and show the plot
    plt.tight_layout()  # Automatically adjust subplot spacing
    plt.show()


def plot_training_curves(model):
    # Compute all metrics
    metrics = model.val_metrics_tracker.compute_all()

    # Extract relevant metrics
    mse = metrics['MeanSquaredError']
    r2 = metrics['R2Score']

    # Create figure and axis
    fig, ax1 = plt.subplots()

    # Plot MSE on the primary y-axis
    ax1.plot(range(1, len(mse) + 1), mse, marker='o', color='blue', label='MSE')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis for R2Score
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(r2) + 1), r2, marker='^', color='green', label='R2Score')
    ax2.set_ylabel('R2 Score', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Add a title and grid
    plt.title('Metrics (MSE and R2) over Epochs')
    ax1.grid()

    # Add legend manually
    fig.legend(loc='center', ncol=2)

    plt.show()
