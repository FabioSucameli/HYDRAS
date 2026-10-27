# Atomic, pickle-free checkpoints for the existing IPP experiment.

from dataclasses import asdict
import json
import os
from types import SimpleNamespace

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from pollutant_gp.model import build_kernel
from pollutant_gp.types import ReconstructionResult


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, default=json_default, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_checkpoint(path, raw, name, diagnostic, theta, optimizer_rows, **state):
    metadata = dict(
        name=name, diagnostic=dict(
            final_lml=diagnostic.final_lml, target_mean=diagnostic.target_mean,
            target_scale=diagnostic.target_scale,
            hyperparameters=[dict(name=p.name, optimized_value=p.optimized_value)
                             for p in diagnostic.hyperparameters]),
        optimizer_rows=optimizer_rows, **state)
    arrays = dict(theta=theta)
    if raw is not None:
        fields = asdict(raw)
        arrays.update(mean_field=fields.pop("mean_field"), std_field=fields.pop("std_field"))
        metadata["reconstruction"] = fields
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, default=json_default), **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as saved:
        state = json.loads(str(saved["metadata"]))
        state["theta"] = saved["theta"]
        state["raw"] = (ReconstructionResult(mean_field=saved["mean_field"], std_field=saved["std_field"],
                                              **state.pop("reconstruction"))
                        if "reconstruction" in state else None)
    diagnostic = state["diagnostic"]
    diagnostic["hyperparameters"] = [SimpleNamespace(**p) for p in diagnostic["hyperparameters"]]
    state["diagnostic"] = SimpleNamespace(**diagnostic)
    return state


def restore_model(args, theta, coordinates, values, diagnostic):
    # Only needed after resuming a round with no new cells: condition, never reoptimize.
    kernel = build_kernel(2, "anisotropic", args.length_scale_lower_bound, args.length_scale_upper_bound,
                          args.noise_level_initial, args.noise_level_lower_bound, args.noise_level_upper_bound,
                          constant_value_initial=.5, length_scale_initial=1., second_length_scale_initial=.1)
    model = GaussianProcessRegressor(kernel=kernel.clone_with_theta(theta), alpha=1e-10,
                                     optimizer=None, normalize_y=True).fit(coordinates, values)
    np.testing.assert_allclose([model._y_train_mean, model._y_train_std],
                               [diagnostic.target_mean, diagnostic.target_scale], rtol=1e-12, atol=1e-14)
    return model
