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
        
        # validation outputs and targets
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
        output = self.forward(x)
        loss = self.loss(output, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss(output, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.validation_step_outputs.append(output)
        self.validation_step_targets.append(y)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss(output, y)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        
        # Store predictions, targets
        self.test_step_outputs.append(output)
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
