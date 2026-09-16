from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import os
import torch
import pandas as pd
from datetime import datetime
import pprint
from collections import OrderedDict

class GeneratorWrapper:
    def __init__(self, model_cls, model_params, train_params, pipeline):
        self.model_cls = model_cls
        self.model_params = model_params
        self.train_params = train_params
        self.pipeline = pipeline
        self.model = None
        self.feature_columns = None


    def fit(self, train_df):
        X_train = self.pipeline.transform(train_df)

        self.feature_columns = list(X_train.columns)
        X_train_np = X_train.values.astype("float32")

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_np, dtype=torch.float32)),
            batch_size=self.train_params.get("batch_size", 64),
            shuffle=True
        )

        # store the true final model params
        self.model_params = dict(self.model_params)
        self.model_params["input_dim"] = X_train_np.shape[1]

        self.model = self.model_cls(**self.model_params)

        self.model.fit(
            train_loader=train_loader,
            val_loader=None,
            **self.train_params
        )

    def generate(self, n_samples):
        X_synth = self.model.generate(n_samples)

        X_synth = pd.DataFrame(
            X_synth.values,
            columns=self.feature_columns
        )

        return self.pipeline.inverse_transform(X_synth)



def save_generator(path, generator, metadata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "saved_at": datetime.utcnow().isoformat(),
        "model_params": generator.model_params,
        "train_params": generator.train_params,
        "model_state_dict": generator.model.model.state_dict(),
        "pipeline_state": generator.pipeline.__dict__.copy(),
        "feature_columns": list(generator.feature_columns),
        "metadata": metadata or {},
    }

    torch.save(ckpt, path)
    print(f"Saved generator to: {path}")



def load_generator(path, model_cls, pipeline_cls, generator_cls=GeneratorWrapper, map_location="cpu", print_ckpt = True):


    def truncate_value(value, max_len=120):

        # tensors / huge arrays
        if hasattr(value, "shape"):
            return f"<{type(value).__name__} shape={tuple(value.shape)}>"

        # very long strings
        if isinstance(value, str):
            return value if len(value) <= max_len else value[:max_len] + "..."

        # dict-like
        if isinstance(value, dict):
            return {
                k: truncate_value(v, max_len=max_len)
                for k, v in value.items()
            }

        # lists / tuples
        if isinstance(value, (list, tuple)):
            shortened = [
                truncate_value(v, max_len=max_len)
                for v in value[:10]
            ]

            if len(value) > 10:
                shortened.append(f"... ({len(value)-10} more items)")

            return shortened

        # OrderedDict (e.g. state_dict)
        if isinstance(value, OrderedDict):
            return f"<OrderedDict with {len(value)} items>"

        # generic repr truncation
        text = repr(value)

        if len(text) > max_len:
            return text[:max_len] + "..."

        return value



    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    pipeline = pipeline_cls.__new__(pipeline_cls)
    pipeline.__dict__.update(ckpt["pipeline_state"])

    model = model_cls(**ckpt["model_params"])
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.model.eval()

    generator = generator_cls(
        model_cls=model_cls,
        model_params=ckpt["model_params"],
        train_params=ckpt["train_params"],
        pipeline=pipeline
    )

    generator.model = model
    generator.feature_columns = ckpt["feature_columns"]


    if print_ckpt:
        summary = {
            k: truncate_value(v)
            for k, v in ckpt.items()
            if k != "model_state_dict"
        }
        if "model_state_dict" in ckpt:
            summary["model_state_dict"] = (
                f"<state_dict with {len(ckpt['model_state_dict'])} tensors>"
            )
        pprint.pprint(summary, sort_dicts=False)


    return generator, ckpt
