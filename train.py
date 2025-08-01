from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch

# Import transforms from torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf

# Import your refactored modules
from src.noisy_ood.data import ChestXrayDataModule
from src.noisy_ood.model import ResNet50FineTune, ResNet50LightningModule
from src.noisy_ood.transforms import NoiseAugmentation


def build_transforms(cfg: DictConfig):
    # Validation and Test transforms are mostly fixed
    # no noise val transform
    val_no_noise = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # predict
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Training transform depends on the config
    if cfg.experiment.apply_train_noise:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomChoice([
                NoiseAugmentation(mode='gaussian', mean=cfg.noise_params.mean, var=cfg.noise_params.variance),
                NoiseAugmentation(mode='s&p', density=cfg.noise_params.density),
                NoiseAugmentation(mode='speckle', var=cfg.noise_params.variance),
                NoiseAugmentation(mode='poisson'),
                transforms.Lambda(lambda x: x) # no noise
            ]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip()
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip()
        ])

    # Decide which validation transforms to use
    val_transforms = [val_no_noise]

    return {
        "train_transform": train_transform,
        "val_transform_list": val_transforms,
        "test_transform": test_transform
    }


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Mixed precision training
    torch.set_float32_matmul_precision('medium')
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    all_experiment_results = []

    for seed in cfg.training.seeds:
        pl.seed_everything(seed, workers=True)
        experiment_name = f"{cfg.experiment.name}_seed_{seed}"
        print(f"--- Starting Experiment: {experiment_name} ---")

        # 1. Build Transforms and Data Paths from config
        transform_pipeline = build_transforms(cfg)
        
        val_data_paths = [cfg.data.val_csv] * len(transform_pipeline['val_transform_list'])
        test_data_paths = [cfg.data.seen_test_csv, cfg.data.unseen_test_csv]
        
        # ... create val_idx_to_name and test_idx_to_name maps ...
        val_idx_to_name = {
            0: "val_no_noise",
        }

        test_idx_to_name = {
            0: "test_same_distribution",
            1: "test_different_distribution"
        }

        # 2. Setup DataModule
        datamodule = ChestXrayDataModule(
            train_df=cfg.data.train_csv,
            val_df_list=val_data_paths,
            test_df_list=test_data_paths,
            transform_dict=transform_pipeline,
            batch_size=cfg.training.batch_size,
        )

        # 3. Setup Model
        model_arch = ResNet50FineTune()
        lightning_model = ResNet50LightningModule(
            model=model_arch,
            val_idx_to_name=val_idx_to_name,
            test_idx_to_name=test_idx_to_name,
            learning_rate=cfg.training.lr
        )

        # 4. Setup Trainer (Callbacks, Logger)
        checkpoints_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"{cfg.callbacks.checkpoints.dirpath}/seed_{seed}",
            filename="best_model-{epoch:02d}-{val_avg_auc:.4f}",
            monitor=cfg.callbacks.checkpoints.monitor,
            mode=cfg.callbacks.checkpoints.mode,
            save_top_k=cfg.callbacks.checkpoints.save_top_k,
            verbose=cfg.callbacks.checkpoints.verbose,
        )

        earlystopping_callback = pl.callbacks.EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            mode=cfg.callbacks.early_stopping.mode,
            patience=cfg.callbacks.early_stopping.patience,
            min_delta=cfg.callbacks.early_stopping.min_delta,
            stopping_threshold=cfg.callbacks.early_stopping.stopping_threshold
        )

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=cfg.training.device_num,
            logger=pl.loggers.TensorBoardLogger(Path(cfg.training.log_dir) / "runs", name=f"experiment_{experiment_name}"),
            callbacks=[checkpoints_callback, earlystopping_callback],
            log_every_n_steps=cfg.training.log_every_n_steps,
        )

        # 5. Run Training and Testing
        print(f"{seed} - Starting Training...")
        trainer.fit(model=lightning_model, datamodule=datamodule)
        print(f"{seed} - Training Complete. Testing...")
        trainer.test(ckpt_path="best", datamodule=datamodule, verbose=False)

        # Save the logged metrics for each seed
        run_summary = {k: v.item() for k, v in trainer.logged_metrics.items()}
        run_summary['seed'] = seed
        all_experiment_results.append(run_summary)
    
    print("--- All Experiments Completed ---")
    
    results_df = pd.DataFrame(all_experiment_results)
    metric_cols_same_dist = [col for col in results_df.columns if 'seed' not in col and 'loss' not in col and 'same' in col]
    metric_cols_diff_dist = [col for col in results_df.columns if 'seed' not in col and 'loss' not in col and 'different' in col]
    summary_stats_same_dist = results_df[metric_cols_same_dist].agg(['mean'])
    summary_stats_diff_dist = results_df[metric_cols_diff_dist].agg(['mean'])

    summary_stats_same_dist.columns = [col.split('/')[0] for col in summary_stats_same_dist.columns]
    summary_stats_diff_dist.columns = [col.split('/')[0] for col in summary_stats_diff_dist.columns]

    print("\n--- Summary Statistics Across All Seeds For In-Distribution Evaluation ---")
    print(summary_stats_same_dist)

    print("\n--- Summary Statistics Across All Seeds For Out-of-Distribution Evaluation ---")
    print(summary_stats_diff_dist)

    print("\n--- Difference Between In-Distribution and Out-of-Distribution Evaluation ---")
    summary_stats_diff = summary_stats_same_dist.subtract(summary_stats_diff_dist)
    summary_stats_diff = summary_stats_diff.abs()
    print(summary_stats_diff)

    print("\n--- Saving Summary Statistics ---")
    results_df.to_csv(f"{cfg.summary_save_dir}/summary_raw_results.csv",float_format="%.4f")
    summary_stats_same_dist.to_csv(f"{cfg.summary_save_dir}/summary_statistics_id.csv", float_format="%.4f")
    summary_stats_diff_dist.to_csv(f"{cfg.summary_save_dir}/summary_statistics_ood.csv", float_format="%.4f")
    summary_stats_diff.to_csv(f"{cfg.summary_save_dir}/summary_statistics_diff.csv", float_format="%.4f")
       
if __name__ == "__main__":
    main()