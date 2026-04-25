import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from .models import MLPDiffusionModel
from .diffusion import GaussianDiffusion


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    import numpy as np, random

    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TabDDPMGenerator:
    def __init__(self, params):
        self.device = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = params.get("hidden_dim", 128)
        self.num_layers = params.get("num_layers", 3)
        self.time_embed_dim = params.get("time_embed_dim", 64)
        self.timesteps = params.get("timesteps", 1000)
        self.schedule = params.get("schedule", "linear")
        self.beta_scale = params.get("beta_scale", 1.0)
        self.dropout = params.get("dropout", 0.0)
        self.lr = params.get("lr", 1e-3)
        self.batch_size = params.get("batch_size", 128)
        self.epochs = params.get("epochs", 1000)
        self.seed = int(params.get("seed", 0))
        self.include_label = bool(params.get("include_label", False))
        self.lambda_cat = float(params.get("lambda_cat", 0.5))
        self.agent_id = str(params.get("agent_id", "tabddpm"))
        self.feature_alpha = (
            torch.as_tensor(params.get("feature_alpha"), dtype=torch.float32)
            if params.get("feature_alpha") is not None
            else None
        )
        self.feature_gamma = (
            torch.as_tensor(params.get("feature_gamma"), dtype=torch.float32)
            if params.get("feature_gamma") is not None
            else None
        )

        self.continuous_idx = list(params.get("continuous_idx", []))
        self.categorical_idx = list(params.get("categorical_idx", []))
        self.categorical_cardinalities = list(params.get("categorical_cardinalities", []))
        fw_cont = params.get("feature_weights_cont", None)
        fw_cat = params.get("feature_weights_cat", None)
        self.feature_weights_cont = (
            torch.as_tensor(fw_cont, dtype=torch.float32) if fw_cont is not None else None
        )
        self.feature_weights_cat = (
            torch.as_tensor(fw_cat, dtype=torch.float32) if fw_cat is not None else None
        )
        # DP parameters
        self.epsilon = params.get("epsilon", None)
        self.delta = params.get("delta", 1e-5)
        self.max_grad_norm = params.get("max_grad_norm", 1.0)
        
        self.model = None
        self.diffusion = None
        self.privacy_engine = None
        self.label_values = None

        print(f"[TabDDPM] Using device: {self.device}")
        if self.epsilon:
            print(f"[TabDDPM] DP enabled with epsilon={self.epsilon}")

    # -----------------------------------------------------
    def set_per_feature_schedule(
        self,
        feature_alpha: np.ndarray | torch.Tensor,
        feature_gamma: np.ndarray | torch.Tensor,
        agent_id: str | None = None,
    ) -> None:
        self.feature_alpha = torch.as_tensor(feature_alpha, dtype=torch.float32)
        self.feature_gamma = torch.as_tensor(feature_gamma, dtype=torch.float32)
        if agent_id is not None:
            self.agent_id = str(agent_id)

    # -----------------------------------------------------
    def _build_model(self, input_dim):
        self.model = MLPDiffusionModel(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            depth=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        x_mean = np.asarray(getattr(self, "X_mean", np.zeros(input_dim, dtype=float)), dtype=float)
        x_std = np.asarray(getattr(self, "X_std", np.ones(input_dim, dtype=float)), dtype=float)

        self.diffusion = GaussianDiffusion(
            model=self.model,
            timesteps=self.timesteps,
            device=self.device,
            schedule=self.schedule,
            beta_scale=self.beta_scale,
            continuous_idx=self.continuous_idx,
            categorical_idx=self.categorical_idx,
            categorical_cardinalities=self.categorical_cardinalities,
            feature_weights_cont=self.feature_weights_cont,
            feature_weights_cat=self.feature_weights_cat,
            lambda_cat=self.lambda_cat,
            x_mean=torch.as_tensor(x_mean, dtype=torch.float32),
            x_std=torch.as_tensor(x_std, dtype=torch.float32),
            feature_alpha=self.feature_alpha,
            feature_gamma=self.feature_gamma,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    # -----------------------------------------------------
    def _compute_noise_multiplier(self, dataset_size):
        """Compute noise multiplier for DP training."""
        if dataset_size <= 0:
            raise ValueError("Dataset is empty. Provide a non-empty dataset before enabling DP training.")

        sample_rate = self.batch_size / dataset_size
        
        # Validate sample rate
        if sample_rate >= 1.0:
            raise ValueError(
                f"Sample rate q={sample_rate:.4f} is invalid (batch_size={self.batch_size}, dataset_size={dataset_size}). "
                "Reduce --bsz or increase dataset size so q < 1.0 before enabling DP."
            )
        
        # Simple noise multiplier calculation (can be improved)
        noise_multiplier = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return noise_multiplier
    
    def _train_model(self, X):
        dataset = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Adjust batch size if needed for DP
        if self.epsilon and self.batch_size * 10 > len(dataset):
            self.batch_size = max(1, len(dataset) // 10)
            print(f"Adjusted batch size to {self.batch_size} for DP training")
        
        g = torch.Generator()
        g.manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        if len(loader) == 0:
            raise ValueError(
                f"DataLoader produced zero batches (dataset_size={len(dataset)}, batch_size={self.batch_size}). "
                "Reduce --bsz or increase dataset size so DP sampling rate is well-defined."
            )

        self.model.train()

        # Setup DP training if epsilon is specified
        if self.epsilon:
            noise_multiplier = self._compute_noise_multiplier(len(dataset))
            self.privacy_engine = PrivacyEngine()
            
            print(f"\n=== DP SETUP VERIFICATION ===")
            print(f"Dataset size: {len(dataset)}")
            print(f"Batch size: {self.batch_size}")
            print(f"Sample rate q: {self.batch_size / len(dataset):.4f}")
            print(f"Target epsilon: {self.epsilon}")
            print(f"Target delta: {self.delta}")
            print(f"Max grad norm: {self.max_grad_norm}")
            
            self.model, self.optimizer, loader = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=loader,
                epochs=self.epochs,
                target_epsilon=self.epsilon,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )
            
            # Print actual DP parameters after setup
            try:
                actual_noise_multiplier = self.privacy_engine.noise_multiplier
                actual_sample_rate = self.privacy_engine.sample_rate
                actual_max_grad_norm = self.privacy_engine.max_grad_norm
                print(f"Actual noise multiplier: {actual_noise_multiplier:.4f}")
                print(f"Actual sample rate: {actual_sample_rate:.4f}")
                print(f"Actual max grad norm: {actual_max_grad_norm}")
            except AttributeError:
                # Fallback for different Opacus versions
                print(f"DP parameters set (exact values not accessible in this Opacus version)")
            print(f"================================\n")

        for epoch in range(self.epochs):
            if self.epsilon:
                # DP training with batch memory manager
                with BatchMemoryManager(
                    data_loader=loader,
                    max_physical_batch_size=self.batch_size,
                    optimizer=self.optimizer
                ) as memory_safe_data_loader:
                    for batch in memory_safe_data_loader:
                        batch = batch.float()
                        loss = self.diffusion.training_loss(batch)
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
            else:
                # Standard training
                for batch in loader:
                    batch = batch.float()
                    loss = self.diffusion.training_loss(batch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        
        # Print final privacy accounting
        if self.epsilon:
            final_epsilon = self.privacy_engine.accountant.get_epsilon(delta=self.delta)
            print(f"\n=== FINAL PRIVACY ACCOUNTING ===")
            print(f"Target epsilon: {self.epsilon}")
            print(f"Achieved epsilon: {final_epsilon:.4f}")
            print(f"Delta: {self.delta}")
            print(f"Total steps: {self.epochs * len(loader)}")
            print(f"================================\n")

    # -----------------------------------------------------
    def fit(self, X_real, y_real=None):
        """Fit the model to real data."""
        # Convert to numpy if needed
        if hasattr(X_real, "values"):
            X_real = X_real.values
        if y_real is not None and hasattr(y_real, "values"):
            y_real = y_real.values

        X_real = np.asarray(X_real)
        if self.include_label:
            # Treat label as the last column in X_real
            self.label_values = np.unique(X_real[:, -1])
        else:
            self.y_real = y_real

        # Normalize input
        self.X_mean = X_real.mean(axis=0)
        self.X_std = X_real.std(axis=0) + 1e-6
        X_norm = (X_real - self.X_mean) / self.X_std

        # Store input dimension
        self.input_dim = X_norm.shape[1]

        # Build model
        if self.model is None:
            self._build_model(self.input_dim)

        # Train model
        self._train_model(X_norm)
    
    def generate(self, n_samples, scale=1.0, shift=0.0):
        """Generate synthetic samples."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Generate samples
        with torch.no_grad():
            samples = self.diffusion.sample((n_samples, self.input_dim))
            samples = samples.cpu().numpy()
        
        # Denormalize
        X_syn = samples * self.X_std + self.X_mean

        # Apply scale & shift
        X_syn = X_syn * scale + shift

        # Replace NaNs/Infs with mean
        if np.isnan(X_syn).any() or np.isinf(X_syn).any():
            X_syn = np.nan_to_num(X_syn, nan=0.0, posinf=0.0, neginf=0.0)

        if self.include_label:
            if X_syn.shape[1] < 1:
                raise ValueError("Joint diffusion output has no columns.")
            y_raw = X_syn[:, -1]
            X_syn = X_syn[:, :-1]
            if self.label_values is None or len(self.label_values) == 0:
                raise ValueError("Label values not initialized for joint diffusion.")
            # Map to nearest label value
            y_syn = np.array(
                [self.label_values[np.argmin(np.abs(self.label_values - val))] for val in y_raw]
            )
        else:
            # Sample labels from real distribution
            y_syn = np.random.choice(self.y_real, size=n_samples)

        return X_syn, y_syn
    
    def generate_legacy(self, X_real, y_real, scale, shift):

        # Normalize input
        X_norm = (X_real - X_real.mean(axis=0)) / (X_real.std(axis=0) + 1e-6)

        # Build model only once
        if self.model is None:
            self._build_model(X_norm.shape[1])

        # Train diffusion model
        self._train_model(X_norm)

        # Generate samples
        with torch.no_grad():
            samples = self.diffusion.sample((len(X_real), X_norm.shape[1]))
            samples = samples.cpu().numpy()

        # Denormalize
        X_syn = samples * X_real.std(axis=0) + X_real.mean(axis=0)

        # Apply scale & shift
        X_syn = X_syn * scale + shift

        # Sample labels from real distribution
        y_syn = np.random.choice(y_real, size=len(y_real))

        return X_syn, y_syn
