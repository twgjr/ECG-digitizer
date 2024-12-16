import torch
from torch import nn, optim
import torchmetrics as TM
import lightning as L


class ECGRegressor(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.loss = nn.MSELoss(reduction='sum')
        self.encoder = encoder
        self.decoder = decoder
        
        # Validation metrics
        self.val_metrics_tracker = TM.wrappers.MetricTracker(TM.MetricCollection([
            TM.regression.MeanSquaredError(), 
            TM.regression.MeanAbsoluteError(), 
            TM.regression.R2Score()
        ]))
        self.validation_step_outputs = []
        self.validation_step_targets = []

        # Test metrics
        self.test_metrics_tracker = TM.wrappers.MetricTracker(TM.MetricCollection([
            TM.regression.MeanSquaredError(), 
            TM.regression.MeanAbsoluteError(), 
            TM.regression.R2Score()
        ]))
        
        # Test outputs and targets
        self.test_step_outputs = []
        self.test_step_targets = []

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.validation_step_outputs.append(logits)
        self.validation_step_targets.append(y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        
        # Store predictions, targets, and residuals
        self.test_step_outputs.append(logits)
        self.test_step_targets.append(y)
        return loss
    
    def on_validation_start(self):
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()
    
    def on_validation_epoch_end(self):
        all_preds = torch.vstack(self.validation_step_outputs)
        all_targets = torch.vstack(self.validation_step_targets)
        loss = nn.functional.mse_loss(all_preds, all_targets)
        self.val_metrics_tracker.increment()
        self.val_metrics_tracker.update(all_preds, all_targets)
        self.log('val_loss_epoch_end', loss)
    
    def on_test_epoch_end(self):
        all_preds = torch.vstack(self.test_step_outputs)
        all_targets = torch.vstack(self.test_step_targets)
        
        self.test_metrics_tracker.increment()
        self.test_metrics_tracker.update(all_preds, all_targets)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class CNNModel(nn.Module):
    def __init__(self, image_dims=(425, 10), num_channels=1, num_kernels=64, kernel_size=3, stride=1, padding=1, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, num_kernels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_kernels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves spatial dimensions
            
            nn.Conv2d(num_kernels, num_kernels * 2, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_kernels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(num_kernels * 2, num_kernels * 4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_kernels * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Flatten and project to scalar
        self.encoder_output_dim = num_kernels * 4  # Matches final channel count
        self.embedding_layer = nn.Sequential(
            nn.Flatten(),  # Flattens (256, 53, 1)
            nn.Linear(self.encoder_output_dim * 53 * 1, self.output_dim)  # Output is 1 scalar
        )

    def forward(self, x):
        x = self.encoder(x)  # Pass through the encoder
        x = self.embedding_layer(x)  # Pass through the embedding layer
        return x
