# Intro
* Started with Fall 2024 project based on the PhysioNet Challenge 2024. Since then it has finsihed, published and is in late submissions on Kaggle.
* The dataset is much more readily available that during the challenge where I had to generate synthetic ECG images. Now there is over 80GB of ECG images and signal data available for download from Kaggle.
* I have taken it as a challenge to turn this into a full production ready pipeline and efficient models.

# Plan
The approach I am taking is to use a multi-stage training process which is typically called 
I plan to train the model in three stages:
1. Train a signal autocoder to learn a good representation of the ECG signals without the images or labels. 1D to 1D CNN.
2. Freeze the decoder and train an image encoder to learn to map the ECG images to the same latent space as the signal autocoder. 2D to 1D CNN.
3. Unfreeze the decoder and train the whole thing end to end with the labels. 2D to 1D CNN with classification head.

# Data Management

## Restore data
If you tinker with the raw data and need to restore it (e.g. to retest the EDA notebook), use:
```bash
dvc pull data.dvc
```
Note: `dvc pull` alone will fail because `models/signal_autoencoder.ckpt` has no hash recorded (the training pipeline has not been run yet and there is no `dvc.lock`). Use `dvc pull data.dvc` to pull only the data.

## Save changes to data
After modifying dataset files, re-hash, push, and commit:
```bash
dvc add data        # re-hashes data/, updates data.dvc
dvc push            # uploads changed files to Azure
git add data.dvc && git commit -m "describe your changes"
```

## Verify remote is up to date
```bash
dvc status --cloud
```
If the output says `Data and pipelines are up to date.` the remote is in sync. Any files listed need to be pushed.