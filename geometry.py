"""
Fault mesh geometry generation
"""

import numpy as np


def create_fault_mesh(config):
    """
    Create a regular grid for vertical strike-slip fault

    Returns:
    --------
    mesh : dict
        'centroids': (n_elements, 3) array of [x, y, z] positions
        'x_coords': (n_along_strike, n_down_dip) grid
        'z_coords': (n_along_strike, n_down_dip) grid
        'element_areas': (n_elements,) array of areas
        'neighbors': dict of neighbor indices for each element
    """
    # Generate regular grid
    x = np.linspace(0, config.fault_length_km, config.n_along_strike)
    z = np.linspace(0, config.fault_depth_km, config.n_down_dip)

    X, Z = np.meshgrid(x, z, indexing="ij")

    # Centroids (flatten for element-wise operations)
    centroids = np.column_stack(
        [
            X.ravel(),  # x (along-strike)
            np.zeros(X.size),  # y (always 0 for vertical fault)
            Z.ravel(),  # z (depth, positive down)
        ]
    )

    # Element areas (all equal for regular grid)
    element_areas = np.full(config.n_elements, config.element_area_m2)

    # Neighbor connectivity (for slip generation)
    neighbors = compute_neighbors(config)

    mesh = {
        "centroids": centroids,
        "x_coords": X,
        "z_coords": Z,
        "element_areas": element_areas,
        "neighbors": neighbors,
        "n_along_strike": config.n_along_strike,
        "n_down_dip": config.n_down_dip,
    }

    return mesh


def compute_neighbors(config):
    """
    Compute neighbor indices for each element

    Returns dict: {element_idx: [list of neighbor indices]}
    """
    neighbors = {}

    for i in range(config.n_elements):
        # Convert flat index to 2D
        ix = i // config.n_down_dip
        iz = i % config.n_down_dip

        neighbor_list = []

        # Check all 4 directions (N, S, E, W)
        for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, nz = ix + dx, iz + dz

            # Check bounds
            if 0 <= nx < config.n_along_strike and 0 <= nz < config.n_down_dip:
                neighbor_idx = nx * config.n_down_dip + nz
                neighbor_list.append(neighbor_idx)

        neighbors[i] = neighbor_list

    return neighbors


def get_element_index(ix, iz, config):
    """Convert 2D grid coordinates to flat element index"""
    return ix * config.n_down_dip + iz


def get_2d_indices(element_idx, config):
    """Convert flat element index to 2D grid coordinates"""
    ix = element_idx // config.n_down_dip
    iz = element_idx % config.n_down_dip
    return ix, iz
