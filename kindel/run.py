import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import os
from typing import List, Optional

import wandb
import yaml
from pytorch_lightning.loggers import WandbLogger
from redun import Dir, File, task

from kindel.models.basic import RandomForest, KNeareastNeighbors, XGBoost
from kindel.models.gnn import GraphIsomorphismNetwork
from kindel.models.torch import DeepNeuralNetwork
from kindel.utils.data import (
    get_training_data,
    get_testing_data,
    kendall,
    spearman,
    rmse,
)
from kindel.utils.helpers import set_seed
from kindel.models.compose import DELCompose
from kindel.utils.plots import plot_regression_metrics

redun_namespace = "kindel"

api = wandb.Api()
BATCH_ENV_VAR_DICT = {
    "containerProperties": {
        "environment": [
            {"name": "", "value": api.client.app_url},
            {"name": "", "value": api.api_key},
        ]
    }
}


def get_model(model_name: str, hyperparameters: dict, wandb_logger: WandbLogger):
    if model_name.lower().startswith("xgboost"):
        model = XGBoost(**hyperparameters)
    elif model_name.lower().startswith("rf"):
        model = RandomForest(**hyperparameters)
    elif model_name.lower().startswith("knn"):
        model = KNeareastNeighbors(**hyperparameters)
    elif model_name.lower().startswith("dnn"):
        model = DeepNeuralNetwork(wandb_logger, **hyperparameters)
    elif model_name.lower().startswith("gin"):
        model = GraphIsomorphismNetwork(wandb_logger, **hyperparameters)
    elif model_name.lower().startswith("compose"):
        model = DELCompose(wandb_logger, **hyperparameters)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

#@task(executor="main", vcpus=5, job_def_extra=BATCH_ENV_VAR_DICT)
@task(executor="main")
def training_subjob(
    model_name: str,
    output_dir: Dir,
    split_index: int,
    split_type: str,
    target: str,
    hyperparameters: Optional[dict] = None,
) -> File:
    # Create directories before starting
    results_dir = os.path.join(output_dir.path, model_name, f"results_{split_type}")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    set_seed(123)
    print("########################################################")
    print("Loading data...")
    print("########################################################")
   
    df_train, df_valid, df_test = get_training_data(
        target, split_index=split_index, split_type=split_type
    )
    print("########################################################")
    print("Data loaded, starting featurization...")
    print("########################################################")
    if hyperparameters is None:
        hyperparameters = {}

    # Initialize model without wandb logger
    model = get_model(model_name, hyperparameters, wandb_logger=None)
    print("Model initialized, preparing dataset...")
    data = model.prepare_dataset(df_train, df_valid, df_test)
    print("Starting training...")
    model.train()

    results = {}

    # Training set performance
    print("computing training set performance")
    train_preds = model.predict(data.train.x)
    results["train"] = {
        "rho": spearman(train_preds, data.train.y),
        "tau": kendall(train_preds, data.train.y),
        "mse": rmse(train_preds, data.train.y)**2
    }
    print(results["train"])
    
    # Validation set performance
    print("computing validation set performance")
    valid_preds = model.predict(data.valid.x)
    results["valid"] = {
        "rho": spearman(valid_preds, data.valid.y),
        "tau": kendall(valid_preds, data.valid.y),
        "mse": rmse(valid_preds, data.valid.y)**2
    }
    print(results["valid"])
    print("computing internal test set performance")
    preds = model.predict(data.test.x)
    rho, tau = spearman(preds, data.test.y), kendall(preds, data.test.y)
    results["test"] = {"rho": rho, "tau": tau, "mse": rmse(preds, data.test.y)**2}
    print(results["test"])

    print("computing performance on the extended held-out set")
    testing_data = get_testing_data(target)
    X_test_on, y_test_on = model.featurize(testing_data["on"])
    preds_on = model.predict(X_test_on)
    X_test_off, y_test_off = model.featurize(testing_data["off"])
    preds_off = model.predict(X_test_off)

    # Choose metric based on model type
    is_compose = model_name.lower().startswith("compose")
    metric_name = "NLL" if is_compose else "MSE"
    
    # Get appropriate metrics
    if is_compose:
        train_metric = model.train_metrics.get("nll", 0)
        valid_metric = model.valid_metrics.get("nll", 0)
        test_metric = model.test_metrics.get("nll", 0)
    else:
        train_metric = rmse(train_preds, data.train.y)**2
        valid_metric = rmse(valid_preds, data.valid.y)**2
        test_metric = rmse(preds, data.test.y)**2
    
    plot_regression_metrics(
        y_test_on, preds_on,
        y_test_off, preds_off,
        title=f"{target}_predictions",
        output_dir=os.path.join(plots_dir),
        log_wandb=False,
        dataset_type="extended",
        train_metric=train_metric,
        valid_metric=valid_metric,
        test_metric=test_metric,
        metric_name=metric_name
    )

    # For extended held-out set
    results["all"] = {
        "on": {
            "rho": spearman(preds_on, y_test_on),
            "tau": kendall(preds_on, y_test_on)
        },
        "off": {
            "rho": spearman(preds_off, y_test_off),
            "tau": kendall(preds_off, y_test_off)
        }
    }

    print("computing performance on the in-library held-out set")
    testing_data = get_testing_data(target, in_library=True)
    X_test_on, y_test_on = model.featurize(testing_data["on"])
    preds_on = model.predict(X_test_on)
    X_test_off, y_test_off = model.featurize(testing_data["off"])
    preds_off = model.predict(X_test_off)

    plot_regression_metrics(
        y_test_on, preds_on,
        y_test_off, preds_off,
        title=f"{target}_predictions",
        output_dir=os.path.join(plots_dir),
        log_wandb=False,
        dataset_type="in_library",
        train_metric=results["train"]["mse"],
        valid_metric=results["valid"]["mse"],
        test_metric=results["test"]["mse"],
        metric_name="MSE"
    )

    # For in-library held-out set
    results["lib"] = {
        "on": {
            "rho": spearman(preds_on, y_test_on),
            "tau": kendall(preds_on, y_test_on)
        },
        "off": {
            "rho": spearman(preds_off, y_test_off),
            "tau": kendall(preds_off, y_test_off)
        }
    }

    print("saving results")
    return save_results(results, results_dir, split_index, target)

def save_results(results, results_dir, split_index, target):
    # Original binary format
    binary_file = File(
        os.path.join(results_dir, f"results_metrics_s{split_index}_{target}.yml")
    )
    with binary_file.open("w") as fp:
        yaml.dump(results, fp)
    
    # Human readable format
    def convert_value(v):
        if isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        return float(v)
    
    readable_results = {
        key: convert_value(value)
        for key, value in results.items()
    }
    
    readable_file = File(
        os.path.join(results_dir, f"results_metrics_s{split_index}_{target}_readable.yml")
    )
    with readable_file.open("w") as fp:
        yaml.dump(readable_results, fp, default_flow_style=False)
    
    return binary_file

#@task(job_def_extra=BATCH_ENV_VAR_DICT)
@task(executor="main")
def train(
    model: str,
    output_dir: Dir,
    targets: List[str] = ["ddr1", "mapk14"],
    splits: List[str] = ["random", "disynthon"],
    split_indexes: List[int] = [1, 2, 3, 4, 5],
    hyperparameters: Optional[File] = None,
) -> List[File]:
    if hyperparameters is not None:
        with hyperparameters.open("r") as fp:
            hyperparameters = yaml.safe_load(fp)

    train_partial = training_subjob.partial(
        model_name=model,
        output_dir=output_dir,
        hyperparameters=hyperparameters,
    )

    return [
        train_partial(split_index=split_index, target=target, split_type=split_type)
        for split_type in splits
        for target in targets  
        for split_index in split_indexes
    ]
