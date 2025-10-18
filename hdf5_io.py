"""
HDF5 I/O utilities for earthquake simulator

Provides memory-efficient storage with streaming writes and lazy reads.
Replaces pickle-based storage for large simulations.
"""

import h5py
import numpy as np
from pathlib import Path


def create_hdf5_file(filepath, config, mesh):
    """
    Initialize HDF5 file structure for simulation results

    Creates resizable datasets for time series data with chunking and compression.
    Stores static configuration and mesh data.

    Parameters:
    -----------
    filepath : str or Path
        Path to HDF5 file to create
    config : Config
        Simulation configuration object
    mesh : dict
        Fault mesh dictionary

    Returns:
    --------
    h5file : h5py.File
        Open HDF5 file handle (caller must close)
    """
    filepath = Path(filepath)

    # Create file (overwrite if exists)
    h5file = h5py.File(filepath, 'w')

    # Store configuration as attributes
    config_group = h5file.create_group('config')
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        # HDF5 attributes must be simple types
        if value is None:
            continue
        elif isinstance(value, (list, tuple)):
            config_group.attrs[key] = np.array(value)
        else:
            config_group.attrs[key] = value

    # Store mesh data (static, written once)
    mesh_group = h5file.create_group('mesh')
    mesh_group.create_dataset('centroids', data=mesh['centroids'], compression='gzip', compression_opts=4)
    mesh_group.create_dataset('x_coords', data=mesh['x_coords'], compression='gzip', compression_opts=4)
    mesh_group.create_dataset('z_coords', data=mesh['z_coords'], compression='gzip', compression_opts=4)
    mesh_group.attrs['n_along_strike'] = mesh['n_along_strike']
    mesh_group.attrs['n_down_dip'] = mesh['n_down_dip']

    # Create resizable datasets for time series
    # Use chunking: (chunk_size, n_elements) for row-wise access
    n_elements = config.n_elements
    chunk_size = min(1000, config.n_time_steps)  # Balance between write performance and read efficiency

    # Time array
    compression_kwargs = {}
    if config.hdf5_compression > 0:
        compression_kwargs = {'compression': 'gzip', 'compression_opts': config.hdf5_compression}

    h5file.create_dataset(
        'times',
        shape=(0,),
        maxshape=(None,),
        dtype='f8',
        chunks=(chunk_size,),
        **compression_kwargs
    )

    # Moment snapshots: (n_timesteps, n_elements)
    h5file.create_dataset(
        'moment_snapshots',
        shape=(0, n_elements),
        maxshape=(None, n_elements),
        dtype='f8',
        chunks=(chunk_size, n_elements),
        **compression_kwargs
    )

    # Release snapshots: (n_timesteps, n_elements)
    h5file.create_dataset(
        'release_snapshots',
        shape=(0, n_elements),
        maxshape=(None, n_elements),
        dtype='f8',
        chunks=(chunk_size, n_elements),
        **compression_kwargs
    )

    # Event debt history
    h5file.create_dataset(
        'event_debt_history',
        shape=(0,),
        maxshape=(None,),
        dtype='f8',
        chunks=(chunk_size,),
        **compression_kwargs
    )

    # Create variable-length dataset for events
    # Events will be stored as a compound datatype for efficiency
    event_dtype = np.dtype([
        ('time', 'f8'),
        ('magnitude', 'f8'),
        ('M0', 'f8'),
        ('geom_moment', 'f8'),
        ('hypocenter_idx', 'i4'),
        ('hypocenter_x_km', 'f8'),
        ('hypocenter_z_km', 'f8'),
        ('gamma_used', 'f8'),
        ('lambda_t', 'f8'),
    ])

    h5file.create_dataset(
        'events',
        shape=(0,),
        maxshape=(None,),
        dtype=event_dtype,
        chunks=(100,),
        compression='gzip',
        compression_opts=config.hdf5_compression
    )

    # Create ragged array groups for variable-length event data
    # These store ruptured_elements, slip, and components which vary per event
    h5file.create_group('event_arrays')

    # Track current snapshot index
    h5file.attrs['n_snapshots'] = 0
    h5file.attrs['n_events'] = 0

    return h5file


class BufferedHDF5Writer:
    """
    Buffered writer for HDF5 snapshots

    Accumulates snapshots in memory and writes in batches for performance.
    """

    def __init__(self, h5file, buffer_size=5000):
        """
        Initialize buffered writer

        Parameters:
        -----------
        h5file : h5py.File
            Open HDF5 file handle
        buffer_size : int
            Number of snapshots to buffer before writing (default: 5000 = ~200MB RAM)
        """
        self.h5file = h5file
        self.buffer_size = buffer_size

        # Buffers
        self.times_buffer = []
        self.moment_buffer = []
        self.release_buffer = []
        self.debt_buffer = []

    def append(self, time, m_current, m_release_cumulative, event_debt):
        """
        Append snapshot to buffer

        Automatically flushes when buffer is full.
        """
        self.times_buffer.append(time)
        self.moment_buffer.append(m_current.copy())
        self.release_buffer.append(m_release_cumulative.copy())
        self.debt_buffer.append(event_debt)

        # Flush if buffer is full
        if len(self.times_buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """
        Write all buffered snapshots to HDF5
        """
        if len(self.times_buffer) == 0:
            return

        n = self.h5file.attrs['n_snapshots']
        n_new = len(self.times_buffer)

        # Resize datasets
        self.h5file['times'].resize((n + n_new,))
        self.h5file['moment_snapshots'].resize((n + n_new, self.moment_buffer[0].shape[0]))
        self.h5file['release_snapshots'].resize((n + n_new, self.release_buffer[0].shape[0]))
        self.h5file['event_debt_history'].resize((n + n_new,))

        # Write batch
        self.h5file['times'][n:n+n_new] = np.array(self.times_buffer)
        self.h5file['moment_snapshots'][n:n+n_new, :] = np.array(self.moment_buffer)
        self.h5file['release_snapshots'][n:n+n_new, :] = np.array(self.release_buffer)
        self.h5file['event_debt_history'][n:n+n_new] = np.array(self.debt_buffer)

        # Update counter
        self.h5file.attrs['n_snapshots'] = n + n_new

        # Clear buffers
        self.times_buffer = []
        self.moment_buffer = []
        self.release_buffer = []
        self.debt_buffer = []


def append_snapshot(h5file, time, m_current, m_release_cumulative, event_debt):
    """
    Append a single timestep snapshot to HDF5 file (unbuffered - slow)

    DEPRECATED: Use BufferedHDF5Writer instead for better performance.

    Parameters:
    -----------
    h5file : h5py.File
        Open HDF5 file handle
    time : float
        Current simulation time (years)
    m_current : ndarray
        Current moment distribution (m³)
    m_release_cumulative : ndarray
        Cumulative slip release (m)
    event_debt : float
        Current event debt
    """
    n = h5file.attrs['n_snapshots']

    # Resize datasets to accommodate new row
    h5file['times'].resize((n + 1,))
    h5file['moment_snapshots'].resize((n + 1, m_current.shape[0]))
    h5file['release_snapshots'].resize((n + 1, m_release_cumulative.shape[0]))
    h5file['event_debt_history'].resize((n + 1,))

    # Write data
    h5file['times'][n] = time
    h5file['moment_snapshots'][n, :] = m_current
    h5file['release_snapshots'][n, :] = m_release_cumulative
    h5file['event_debt_history'][n] = event_debt

    # Update counter
    h5file.attrs['n_snapshots'] = n + 1


def append_event(h5file, event):
    """
    Append earthquake event to HDF5 file

    Stores fixed-size event properties in compound dataset.
    Stores variable-length arrays (slip, ruptured_elements) separately.

    Parameters:
    -----------
    h5file : h5py.File
        Open HDF5 file handle
    event : dict
        Event dictionary from simulator
    """
    n = h5file.attrs['n_events']

    # Resize events dataset
    h5file['events'].resize((n + 1,))

    # Create record for compound datatype
    record = (
        event['time'],
        event['magnitude'],
        event['M0'],
        event['geom_moment'],
        event['hypocenter_idx'],
        event['hypocenter_x_km'],
        event['hypocenter_z_km'],
        event['gamma_used'],
        event['lambda_t'],
    )

    h5file['events'][n] = record

    # Store variable-length arrays as separate datasets
    # Use event index as dataset name
    event_arrays = h5file['event_arrays']
    event_group = event_arrays.create_group(f'event_{n:06d}')

    # Store arrays
    event_group.create_dataset('ruptured_elements', data=event['ruptured_elements'], compression='gzip', compression_opts=4)
    event_group.create_dataset('slip', data=event['slip'], compression='gzip', compression_opts=4)

    # Store components dictionary as attributes
    for key, value in event['components'].items():
        event_group.attrs[f'component_{key}'] = value

    # Update counter
    h5file.attrs['n_events'] = n + 1


def finalize_simulation(h5file, cumulative_loading, cumulative_release, coupling_history, final_moment, slip_rate):
    """
    Store final simulation state and scalars

    Parameters:
    -----------
    h5file : h5py.File
        Open HDF5 file handle
    cumulative_loading : float
        Total moment loaded (m³)
    cumulative_release : float
        Total moment released (m³)
    coupling_history : list
        Coupling values sampled during simulation
    final_moment : ndarray
        Final moment distribution (m³)
    slip_rate : ndarray
        Slip rate distribution (m/year)
    """
    # Store final scalars as attributes
    h5file.attrs['cumulative_loading'] = cumulative_loading
    h5file.attrs['cumulative_release'] = cumulative_release

    # Store final arrays
    h5file.create_dataset('final_moment', data=final_moment, compression='gzip', compression_opts=4)
    h5file.create_dataset('slip_rate', data=slip_rate, compression='gzip', compression_opts=4)

    # Store coupling history if available
    if coupling_history:
        h5file.create_dataset('coupling_history', data=np.array(coupling_history), compression='gzip', compression_opts=4)


def read_config(h5file):
    """
    Reconstruct Config object from HDF5 attributes

    Parameters:
    -----------
    h5file : h5py.File
        Open HDF5 file handle

    Returns:
    --------
    config : Config
        Reconstructed configuration object
    """
    from config import Config

    config = Config()
    config_group = h5file['config']

    # Load attributes back into config object
    for key in config_group.attrs.keys():
        value = config_group.attrs[key]
        # Convert numpy arrays back to lists for pulse configs
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()  # Scalar
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        setattr(config, key, value)

    return config


def load_lazy_results(filepath):
    """
    Load results with lazy-loading for large datasets

    Returns a dict-like object that loads data on-demand.
    Keeps HDF5 file open for streaming access.

    Parameters:
    -----------
    filepath : str or Path
        Path to HDF5 results file

    Returns:
    --------
    results : HDF5Results
        Lazy-loading results object
    """
    return HDF5Results(filepath)


class HDF5Results:
    """
    Lazy-loading results container

    Provides dict-like interface but loads data on-demand from HDF5.
    Keeps file handle open for efficient access.
    """

    def __init__(self, filepath):
        """
        Open HDF5 file for lazy access

        Parameters:
        -----------
        filepath : str or Path
            Path to HDF5 results file
        """
        self.filepath = Path(filepath)
        self.h5file = h5py.File(filepath, 'r')
        self._config = None
        self._mesh = None
        self._events = None

    def __contains__(self, key):
        """
        Check if key exists in results (for 'in' operator)
        """
        valid_keys = [
            'config', 'mesh', 'event_history', 'moment_snapshots',
            'release_snapshots', 'times', 'event_debt_history',
            'snapshot_times', 'cumulative_loading', 'cumulative_release',
            'final_moment', 'slip_rate', 'coupling_history'
        ]
        return key in valid_keys

    def get(self, key, default=None):
        """
        Get item with default value (dict-like interface)
        """
        try:
            return self[key]
        except KeyError:
            return default

    def __getitem__(self, key):
        """
        Get item from results (loads on-demand)
        """
        if key == 'config':
            if self._config is None:
                self._config = read_config(self.h5file)
            return self._config

        elif key == 'mesh':
            if self._mesh is None:
                mesh_group = self.h5file['mesh']
                self._mesh = {
                    'centroids': mesh_group['centroids'][:],
                    'x_coords': mesh_group['x_coords'][:],
                    'z_coords': mesh_group['z_coords'][:],
                    'n_along_strike': int(mesh_group.attrs['n_along_strike']),
                    'n_down_dip': int(mesh_group.attrs['n_down_dip']),
                }
            return self._mesh

        elif key == 'event_history':
            if self._events is None:
                self._events = self._load_events()
            return self._events

        elif key in ['moment_snapshots', 'release_snapshots', 'times', 'event_debt_history']:
            # Return HDF5 dataset directly for lazy slicing
            return self.h5file[key]

        elif key == 'snapshot_times':
            # Alias for compatibility
            return self.h5file['times']

        elif key == 'cumulative_loading':
            return self.h5file.attrs['cumulative_loading']

        elif key == 'cumulative_release':
            return self.h5file.attrs['cumulative_release']

        elif key == 'final_moment':
            return self.h5file['final_moment'][:]

        elif key == 'slip_rate':
            return self.h5file['slip_rate'][:]

        elif key == 'coupling_history':
            if 'coupling_history' in self.h5file:
                return self.h5file['coupling_history'][:].tolist()
            else:
                return []

        else:
            raise KeyError(f"Unknown key: {key}")

    def _load_events(self):
        """
        Load all events into memory

        Events are small (<1 MB typically), so load fully.
        """
        n_events = self.h5file.attrs['n_events']
        events = []

        event_records = self.h5file['events'][:]
        event_arrays = self.h5file['event_arrays']

        for i in range(n_events):
            event_group = event_arrays[f'event_{i:06d}']

            # Reconstruct components dict from attributes
            components = {}
            for attr_key in event_group.attrs.keys():
                if attr_key.startswith('component_'):
                    key = attr_key.replace('component_', '')
                    components[key] = event_group.attrs[attr_key]

            # Create event dict
            event = {
                'time': float(event_records[i]['time']),
                'magnitude': float(event_records[i]['magnitude']),
                'M0': float(event_records[i]['M0']),
                'geom_moment': float(event_records[i]['geom_moment']),
                'hypocenter_idx': int(event_records[i]['hypocenter_idx']),
                'hypocenter_x_km': float(event_records[i]['hypocenter_x_km']),
                'hypocenter_z_km': float(event_records[i]['hypocenter_z_km']),
                'gamma_used': float(event_records[i]['gamma_used']),
                'lambda_t': float(event_records[i]['lambda_t']),
                'ruptured_elements': event_group['ruptured_elements'][:],
                'slip': event_group['slip'][:],
                'components': components,
            }

            events.append(event)

        return events

    def close(self):
        """Close HDF5 file"""
        self.h5file.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
