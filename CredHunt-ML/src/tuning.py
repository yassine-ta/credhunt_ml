import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.dataset import CredentialDataset
from src.model import CredentialGenerator
import logging
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from functools import partial
import os

def custom_trial_dirname(trial):
    """Custom trial dirname function."""
    return f"trial_{trial.trial_id}"

class TuneableCredentialPipeline:
    def __init__(self, initial_credentials, config=None):
        self.initial_credentials = initial_credentials
        self.chars = sorted(list(set(''.join([c['value'] for c in initial_credentials]))))
        self.chars = ['<PAD>', '<START>', '<END>'] + self.chars
        self.char_to_index = {ch: i for i, ch in enumerate(self.chars)}
        self.index_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        self.types = sorted(list(set([c['type'] for c in initial_credentials])))
        self.type_to_index = {t: i for i, t in enumerate(self.types)}
        self.index_to_type = {i: t for i, t in enumerate(self.types)}
        self.num_types = len(self.types)

        max_length = 128
        self.dataset = CredentialDataset(initial_credentials, self.char_to_index, self.type_to_index, max_length)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.config = config  # Store the config

    def tune_training(self, config):
        embed_dim = config["embed_dim"]
        hidden_dim = config["hidden_dim"]
        num_layers = config["num_layers"]
        batch_size = config["batch_size"]
        learning_rate = config["learning_rate"]
        type_embed_dim = 64  # Fixed type embedding dimension

        model = CredentialGenerator(self.vocab_size, embed_dim, hidden_dim, num_layers, self.num_types, type_embed_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.char_to_index['<PAD>'])

        model.train()
        for epoch in range(10):
            total_loss = 0
            for batch in self.dataloader:
                optimizer.zero_grad()
                
                values = batch['value'].to(self.device)
                types = batch['type'].to(self.device)
                
                output = model(values, types)
                loss = criterion(output.view(-1, self.vocab_size), values.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            epoch_loss = total_loss / len(self.dataloader)
            tune.report({"loss": epoch_loss})

def tune_hyperparameters(credentials, num_samples=10):
    config = {
        "embed_dim": tune.choice([64, 128, 256]),
        "hidden_dim": tune.choice([128, 256, 512]),
        "num_layers": tune.choice([1, 2, 3]),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
    }
        
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )
    
    search_alg = HyperOptSearch(
        metric="loss",
        mode="min"
    )
    
    reporter = tune.CLIReporter(
        parameter_columns=["embed_dim", "hidden_dim", "num_layers", "batch_size", "learning_rate"],
        metric_columns=["loss", "training_iteration"]
    )
    
    tuner = tune.run(
        partial(TuneableCredentialPipeline(credentials).tune_training),
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
        progress_reporter=reporter,
        resources_per_trial={"cpu": 2, "gpu": 0.5 if torch.cuda.is_available() else 0},
        name="credential_tuning",
        storage_path="file:///E:/AXA%20GBS%20-%20Cred%20Hunt/CredHunt-ML/tmp/ray_results",
        trial_dirname_creator=custom_trial_dirname,
    )
    # Get best config
    best_trial = tuner.get_best_trial("loss", "min", "last")
    return best_trial.config