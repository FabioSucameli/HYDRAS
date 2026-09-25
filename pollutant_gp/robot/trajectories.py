# Geometry-only random walks; concentration never controls robot motion.

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label


@dataclass(frozen=True)
class RobotWalk:
    positions: np.ndarray
    cell_indices: np.ndarray
    rejected: np.ndarray
    center: np.ndarray


# Validate the regular Cartesian grid and retain signed spacing for descending axes.
def grid_geometry(grid):
    x, y = grid.x_grid[0], grid.y_grid[:, 0]
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Robot sampling requires at least two cells per axis.")
    dx, dy = x[1] - x[0], y[1] - y[0]
    if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or dx == 0 or dy == 0
            or not np.allclose(np.diff(x), dx) or not np.allclose(np.diff(y), dy)
            or not np.allclose(grid.x_grid, x[None, :])
            or not np.allclose(grid.y_grid, y[:, None])):
        raise ValueError("Robot sampling requires a regular rectilinear grid.")
    return np.array([x[0], y[0]]), np.array([dx, dy])


# Map continuous robot positions to nearest grid cells without extrapolation.
def cell_at(grid, points):
    origin, spacing = grid_geometry(grid)
    ij = np.floor((np.asarray(points) - origin) / spacing + .5).astype(int)
    ny, nx = grid.field.shape
    inside = (ij[..., 0] >= 0) & (ij[..., 0] < nx) & (ij[..., 1] >= 0) & (ij[..., 1] < ny)
    flat = np.clip(ij[..., 1], 0, ny - 1) * nx + np.clip(ij[..., 0], 0, nx - 1)
    return np.where(inside & grid.valid_mask.ravel()[flat], flat, -1)


# Check every crossed cell, including both sides of edges and diagonal corner contacts.
def segment_is_navigable(grid, start, end):
    origin, spacing = grid_geometry(grid)
    a, b = (np.asarray([start, end]) - origin) / spacing
    delta = b - a
    crossings = [0., 1.]
    for axis in range(2):
        if delta[axis] != 0:
            lo, hi = sorted((a[axis], b[axis]))
            edges = np.arange(np.ceil(lo - .5), np.floor(hi - .5) + 1) + .5
            crossings.extend(((edges - a[axis]) / delta[axis]).tolist())
    times = np.unique(np.clip(crossings, 0, 1))
    times = np.concatenate((times, (times[:-1] + times[1:]) / 2))
    points = a + times[:, None] * delta
    ny, nx = grid.field.shape
    # The tiny offsets include all cells touched exactly on a boundary.
    for offset in ((-1e-10, -1e-10), (-1e-10, 1e-10), (1e-10, -1e-10), (1e-10, 1e-10)):
        ij = np.floor(points + .5 + offset).astype(int)
        if np.any((ij[:, 0] < 0) | (ij[:, 0] >= nx) | (ij[:, 1] < 0) | (ij[:, 1] >= ny)):
            return False
        if not np.all(grid.valid_mask[ij[:, 1], ij[:, 0]]):
            return False
    return True


# Select distinct central deployment cells using geometry, never source or peak location.
def initial_positions(grid, n_robots, radius, seed, center=None):
    grid_geometry(grid)
    if n_robots < 1 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Robot count and deployment radius must be positive.")
    components, count = label(grid.valid_mask)
    if not count:
        raise ValueError("No navigable cells.")
    xy = np.column_stack((grid.x_grid.ravel(), grid.y_grid.ravel()))
    if center is None:
        sizes = np.bincount(components.ravel())[1:]
        candidates = np.flatnonzero(components == 1 + np.argmax(sizes))
        centroid = xy[candidates].mean(axis=0)
        central_cell = candidates[np.argmin(np.linalg.norm(xy[candidates] - centroid, axis=1))]
        center = xy[central_cell]
    else:
        center = np.asarray(center, dtype=float)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("Deployment center must be a finite x/y pair.")
        central_cell = int(cell_at(grid, center))
        if central_cell < 0:
            raise ValueError("Deployment center must lie on a valid marine cell.")
    component = components.ravel()[central_cell]
    candidates = np.flatnonzero((components.ravel() == component)
                               & (np.linalg.norm(xy - center, axis=1) <= radius))
    if len(candidates) < n_robots:
        raise ValueError(f"Only {len(candidates)} distinct deployment cells for {n_robots} robots.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    return xy[rng.choice(candidates, n_robots, replace=False)], np.asarray(center)


# Count the initial observation as acquisition one; invalid moves consume budget in place.
def simulate_random_walk(grid, n_robots, n_steps, step_length, radius, seed, center=None):
    if n_steps < 1 or not np.isfinite(step_length) or step_length <= 0:
        raise ValueError("Acquisition count and step length must be positive.")
    starts, center = initial_positions(grid, n_robots, radius, seed, center)
    positions = np.empty((n_steps, n_robots, 2))
    positions[0] = starts
    rejected = np.zeros((n_steps, n_robots), dtype=bool)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    for t in range(1, n_steps):
        angles = rng.uniform(0, 2 * np.pi, n_robots)
        moves = step_length * np.column_stack((np.cos(angles), np.sin(angles)))
        for robot in range(n_robots):
            start = positions[t - 1, robot]
            end = start + moves[robot]
            rejected[t, robot] = not segment_is_navigable(grid, start, end)
            positions[t, robot] = start if rejected[t, robot] else end
    return RobotWalk(positions, cell_at(grid, positions), rejected, center)


# Keep the first noiseless observation of each cell in acquisition order.
def observations_up_to(walk, budget):
    if budget < 1 or budget > walk.cell_indices.size:
        raise ValueError("Observation budget is outside the simulated walk.")
    visits = walk.cell_indices.ravel()[:budget]
    _, first = np.unique(visits, return_index=True)
    return visits[np.sort(first)]


# Share deployment observations with a nested ideal uniform reference without travel limits.
def uniform_reference(grid, starts, budget, seed):
    remaining = np.setdiff1d(np.flatnonzero(grid.valid_mask), starts)
    if budget < len(starts) or budget > len(starts) + len(remaining):
        raise ValueError("Uniform reference budget exceeds the distinct valid cells.")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    return np.concatenate((starts, rng.choice(remaining, budget - len(starts), replace=False)))
