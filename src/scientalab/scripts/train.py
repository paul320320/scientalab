import argparse
import logging
import os

import lightning as L
import polars as pl
from lightning.pytorch import callbacks, loggers

from scientalab import utils
from scientalab.datamodule import pancreas_datamodule
from scientalab.model import invae
from datetime import datetime

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("__name__")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--invariants", type=list[str], default=["celltype"])
    parser.add_argument("--spurious", type=list[str], default=["batch"])
    parser.add_argument("--latent_dim", type=int, default=35)
    parser.add_argument("-mlflow_experiment", type=str, default="inVAE")
    parser.add_argument("--model_dir", type=str, default="run")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--mlflow_uri", type=str, default="http://127.0.0.1:8080")

    args = parser.parse_args()

    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_dir = os.path.join(args.model_dir, now)
    results_dir = os.path.join(args.results_dir, now)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    dm = pancreas_datamodule.PancreasDataModule(data_dir=args.data_dir)
    dm.prepare_data()
    dm.setup(stage="validate")
    model = invae.inVAE(
        input_dim=dm.data_dim,
        latent_dim=args.latent_dim,
        output_dim=dm.data_dim - dm.invariant_dim - dm.spurious_dim,
        invariant_dim=dm.invariant_dim,
        spurious_dim=dm.spurious_dim,
        scheduler="ReduceOnPlateau",
    )

    logger = loggers.MLFlowLogger(experiment_name=args.mlflow_experiment, tracking_uri=args.mlflow_uri)

    trainer_callbacks = [
        callbacks.LearningRateMonitor(logging_interval="epoch"),
        callbacks.EarlyStopping(monitor="val_loss", patience=100),
        callbacks.ModelCheckpoint(
            dirpath=model_dir,
            filename="best_model",
            monitor="val_loss",
        ),
    ]

    trainer = L.Trainer(logger=logger, callbacks=trainer_callbacks)
    trainer.fit(model, datamodule=dm)

    best_model = invae.inVAE.load_from_checkpoint(f"{model_dir}/best_model.ckpt")
    predictions = trainer.predict(model, dm)
    results = pl.DataFrame(predictions)

    results.write_parquet(f"{results_dir}/predictions.parquet")

    silhouette_raw, silhouette_latent, silhouette_output = (
        utils.batch_ASW(results, feature="input", label="celltype", batch="batch"),
        utils.batch_ASW(results, feature="latent_invariant", label="celltype", batch="batch"),
        utils.batch_ASW(results, feature="output", label="celltype", batch="batch"),
    )
    LOGGER.info(f"silhouette scores: raw {silhouette_raw}, latent {silhouette_latent}, output {silhouette_output}")
