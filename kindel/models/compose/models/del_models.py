from itertools import chain

import pytorch_lightning as pl
import torch
import wandb

import kindel.models.compose.utils.eval as eval
from kindel.models.compose.models.count_model import CountModel
from kindel.models.compose.models.enrichment_model import EnrichmentModel
from kindel.utils import plot_regression_metrics
from kindel.utils.data import get_testing_data
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import yaml
from kindel.utils.data import spearman, kendall, rmse


class DELModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.test_step_outputs = []
        self.train_metrics = {}
        self.valid_metrics = {}
        self.test_metrics = {}
        self.all_metrics = {"on": {}, "off": {}}
        self.lib_metrics = {"on": {}, "off": {}}

        input_dim = None
        embed_type = cfg.model.embed_type
        if embed_type == "fps":
            input_dim = cfg.data.fps_dim
        elif embed_type == "gnn":
            input_dim = cfg.model.hidden_dim

        self.enrichment_model = EnrichmentModel(
            cfg=cfg.model, input_dim=input_dim, output_dim=cfg.model.hidden_dim
        )

        self.count_model = CountModel(
            cfg=cfg.model, n_control=cfg.data.n_control, n_target=cfg.data.n_target
        )

        self.save_hyperparameters()

    def run_model(self, batch):
        """
        batch:
            control_counts: [batch_size, n_control] counts tensor
            target_counts: [batch_size, n_target] counts tensor
            pre: [batch_size] counts tensor
        """

        embedding_dict = self.enrichment_model.compute_embeddings(batch)
        z_out = embedding_dict[
            "z"
        ]  # molecule output embedding [batch_size, hidden_dim]

        enrichment_score_dict = self.enrichment_model.compute_enrichments(z_out)
        log_control_scores = enrichment_score_dict[
            "log_control_scores"
        ]  # [batch_size, 1]
        log_target_scores = enrichment_score_dict[
            "log_target_scores"
        ]  # [batch_size, 1]

        zero_probs_dict = self.enrichment_model.compute_zero_probs(z_out)
        control_zero_prob = zero_probs_dict["control_zero_probs"]  # [batch_size, 1]
        target_zero_prob = zero_probs_dict["target_zero_probs"]  # [batch_size, 1]

        # each distribution is [batch_size, 1]
        control_dists, target_dists = self.count_model(
            log_control_scores=log_control_scores,
            log_target_scores=log_target_scores,
            pre=batch["pre"],
            control_p=control_zero_prob,
        )

        control_nll = self.count_model.compute_negative_log_likelihood(
            dists=control_dists,
            zero_prob=None
            if self.cfg.model.output_dist == "poisson"
            else control_zero_prob,
            counts=batch["control_counts"],
        )
        target_nll = self.count_model.compute_negative_log_likelihood(
            dists=target_dists,
            zero_prob=None
            if self.cfg.model.output_dist == "poisson"
            else target_zero_prob,
            counts=batch["target_counts"],
        )

        control_loss = control_nll.mean()
        target_loss = target_nll.mean()

        total_loss = self.cfg.model.beta * control_loss + target_loss

        output_dict = {
            "control_nll": control_nll.mean(dim=1).detach().cpu().numpy(),
            "target_nll": target_nll.mean(dim=1).detach().cpu().numpy(),
            "control_loss": control_loss,
            "target_loss": target_loss,
            "total_loss": total_loss,
            "log_control_scores": log_control_scores,
            "log_target_scores": log_target_scores,
            "control_p": control_zero_prob,
            "target_p": target_zero_prob,
            "control_dists": control_dists,
            "target_dists": target_dists,
        }

        if "attn_probs" in embedding_dict:
            output_dict["attn_probs"] = embedding_dict["attn_probs"]

        return output_dict

    def training_step(self, batch, batch_idx):
        batch_size = batch["batch_size"]

        output_dict = self.run_model(batch)
        control_p = output_dict["control_p"]
        target_p = output_dict["target_p"]

        control_loss = output_dict["control_loss"]
        target_loss = output_dict["target_loss"]
        total_loss = output_dict["total_loss"]

        self.log(
            "train/total_loss",
            total_loss.item(),
            on_step=self.cfg.log_on_step,
            on_epoch=not self.cfg.log_on_step,
            batch_size=batch_size,
        )
        self.log(
            "train/control_loss",
            control_loss.item(),
            on_step=self.cfg.log_on_step,
            on_epoch=not self.cfg.log_on_step,
            batch_size=batch_size,
        )
        self.log(
            "train/target_loss",
            target_loss.item(),
            on_step=self.cfg.log_on_step,
            on_epoch=not self.cfg.log_on_step,
            batch_size=batch_size,
        )
        self.log(
            "train/control_p",
            control_p.mean().item(),
            on_step=self.cfg.log_on_step,
            on_epoch=not self.cfg.log_on_step,
            batch_size=batch_size,
        )
        self.log(
            "train/target_p",
            target_p.mean().item(),
            on_step=self.cfg.log_on_step,
            on_epoch=not self.cfg.log_on_step,
            batch_size=batch_size,
        )

        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_size = batch["batch_size"]

        output_dict = self.run_model(batch)
        control_loss = output_dict["control_loss"]
        target_loss = output_dict["target_loss"]
        total_loss = output_dict["total_loss"]

        self.log(
            "valid_loss",
            total_loss.item(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/control_loss",
            control_loss.item(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/target_loss",
            target_loss.item(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        batch_size = batch["batch_size"]

        output_dict = self.run_model(batch)
        control_loss = output_dict["control_loss"]
        target_loss = output_dict["target_loss"]
        total_loss = output_dict["total_loss"]

        output_dict["smiles"] = batch["smiles"]
        for i in range(self.cfg.data.n_control):
            output_dict["control_%d" % i] = batch["control_counts"][:, i].cpu().numpy()
        for j in range(self.cfg.data.n_target):
            output_dict["target_%d" % j] = batch["target_counts"][:, j].cpu().numpy()

        self.log(
            "test/total_loss",
            total_loss.mean().item(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test/control_loss",
            control_loss.mean().item(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test/target_loss",
            target_loss.mean().item(),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        if "smiles_a" in batch:
            output_dict.update({"smiles_a": list(batch["smiles_a"])})

        if "smiles_b" in batch:
            output_dict.update({"smiles_b": list(batch["smiles_b"])})

        if "smiles_c" in batch:
            output_dict.update({"smiles_c": list(batch["smiles_c"])})

        self.test_step_outputs.append(output_dict)
        return output_dict

    @torch.no_grad()
    def on_test_epoch_end(self):
        test_log_dict = eval.log_data_table(
            self.test_step_outputs,
            n_control=self.cfg.data.n_control,
            n_target=self.cfg.data.n_target,
            output_dist=self.cfg.model.output_dist,
        )
        wandb.log(test_log_dict)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            chain(self.enrichment_model.parameters(), self.count_model.parameters()),
            lr=self.cfg.lr,
        )

        return {"optimizer": optimizer}

    def save_results(self, output_dir, split_index, target):
        # Get metrics from test_step_outputs
        results = {
            "train": self.train_metrics,
            "valid": self.valid_metrics,
            "test": self.test_metrics,
            "all": {},
            "lib": {}
        }
        
        # Save plots and metrics
        for condition in ("on", "off"):
            results["all"][condition] = {
                "rho": self.all_metrics[condition]["rho"],
                "tau": self.all_metrics[condition]["tau"]
            }
            results["lib"][condition] = {
                "rho": self.lib_metrics[condition]["rho"],
                "tau": self.lib_metrics[condition]["tau"]
            }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save binary format
        binary_file = os.path.join(
            output_dir,
            f"results_metrics_s{split_index}_{target}.yml"
        )
        with open(binary_file, 'w') as fp:
            yaml.dump(results, fp)
        
        # Save readable format
        def convert_value(v):
            if isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            return float(v)
        
        readable_results = {
            key: convert_value(value)
            for key, value in results.items()
        }
        
        readable_file = os.path.join(
            output_dir,
            f"results_metrics_s{split_index}_{target}_readable.yml"
        )
        with open(readable_file, 'w') as fp:
            yaml.dump(readable_results, fp, default_flow_style=False)
        
        # Add plotting
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # For extended held-out set
        testing_data = get_testing_data(target)
        X_test_on, y_test_on = self.featurize(testing_data["on"])
        preds_on = self.predict(X_test_on)
        X_test_off, y_test_off = self.featurize(testing_data["off"])
        preds_off = self.predict(X_test_off)
        
        plot_regression_metrics(
            y_test_on, preds_on,
            y_test_off, preds_off,
            title=f"{target}_predictions",
            output_dir=plots_dir,
            dataset_type="extended",
            train_mse=results["train"]["mse"],
            valid_mse=results["valid"]["mse"], 
            test_mse=results["test"]["mse"]
        )
        
        # Same for in-library held-out set...
        
        return binary_file
