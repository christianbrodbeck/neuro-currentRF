"""Forward model and noise covariance with derived whitened quantities."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, TYPE_CHECKING

from eelbrain import NDVar, Sensor, SourceSpace, Space, VolumeSourceSpace
import mne
import numpy as np
from scipy import linalg

from ._initialization import MNEInitializer
from ._linalg import _inv_sqrtm
from ._repr import _forward_summary
from ._typing import FloatArray, NoiseArg

if TYPE_CHECKING:
    from ._data import RegressionData


def _assert_sensors_equal(
        names: Sequence[str],
        reference: Sequence[str],
        desc: str,
        reference_desc: str,
) -> None:
    """Check that two channel lists are identical, including their order.

    Sensor data is combined by channel position throughout, so anything but an
    exact match silently attributes the data of one channel to another.

    Parameters
    ----------
    names, reference
        Channel lists to compare.
    desc, reference_desc
        What the two lists describe, for the error message.
    hint
        Why the channels have to match, appended to the error message.

    Raises
    ------
    ValueError
        If the two lists differ in channels or in their order.
    """
    names, reference = list(names), list(reference)
    if names == reference:
        return
    only_names = [name for name in names if name not in set(reference)]
    only_reference = [name for name in reference if name not in set(names)]
    if only_names or only_reference:
        difference = f"only in {desc}: {only_names or 'none'}; only in {reference_desc}: {only_reference or 'none'}"
    else:
        difference = f"same channels in a different order; {desc} starts with {names[:3]}, {reference_desc} with {reference[:3]}"
    raise ValueError(f"{desc} sensors do not match the {reference_desc} ({difference})")


def _noise_covariance(noise: NoiseArg, sensor: Sensor) -> FloatArray:
    """Sensor-space noise covariance for ``sensor``, in ``sensor``'s channel order.

    Parameters
    ----------
    noise
        Noise as :class:`mne.Covariance`, as :class:`eelbrain.NDVar` data (typically
        an empty-room recording) from which the covariance is estimated, or as a
        covariance matrix for ``sensor``.
    sensor
        Sensors of the lead field. ``noise`` has to have exactly these channels,
        in the same order.

    Raises
    ------
    ValueError
        If ``noise`` does not have exactly ``sensor``'s channels, in ``sensor``'s
        order.
    TypeError
        If ``noise`` is none of the supported types.
    """
    if isinstance(noise, mne.Covariance):
        _assert_sensors_equal(noise.ch_names, sensor.names, 'noise covariance', 'lead field')
        return np.diag(noise.data) if noise['diag'] else noise.data
    elif isinstance(noise, NDVar):
        _assert_sensors_equal(noise.get_dim('sensor').names, sensor.names, 'noise data', 'lead field')
        er = noise.get_data(('sensor', 'time'))
        return np.dot(er, er.T) / er.shape[1]
    elif isinstance(noise, np.ndarray):
        # shape is checked against the lead field in ForwardModel.__post_init__
        return noise
    else:
        raise TypeError(f"Invalid noise type: {type(noise)}. Must be NDVar, mne.Covariance, or ndarray.")


@dataclass(eq=False, repr=False)
class ForwardModel:
    """Forward model and noise covariance with derived whitened quantities.

    The lead field and noise covariance are stored as supplied; the whitened
    quantities used by the solver are derived (and recomputed on unpickling)
    rather than stored.  A single instance is shared read-only across
    cross-validation folds.

    Parameters
    ----------
    lead_field
        Forward solution as a 2-D array, shape ``(n_sensors, n_sources)`` or
        ``(n_sensors, n_sources * len(space))`` for free orientation.
    noise_covariance
        Sensor-space noise covariance, shape ``(n_sensors, n_sensors)``, with the
        channels in the order of ``sensor``.
    source
        Source dimension of the forward model.
    sensor
        Sensor dimension of the forward model.
    space
        Orientation dimension (``None`` for fixed orientation).
    """

    lead_field: FloatArray
    noise_covariance: FloatArray
    source: SourceSpace | VolumeSourceSpace
    sensor: Sensor
    space: Space | None
    #: Inverse square root of :attr:`~ncrf.ForwardModel.noise_covariance` used to whiten sensor data.
    whitening_filter: FloatArray = field(init=False)
    #: Whitened and spectrally normalized :attr:`~ncrf.ForwardModel.lead_field` used by solvers.
    whitened_lead_field: FloatArray = field(init=False)
    #: :attr:`~ncrf.ForwardModel.noise_covariance` transformed by :attr:`~ncrf.ForwardModel.whitening_filter`.
    whitened_noise_covariance: FloatArray = field(init=False)
    #: Spectral norm removed from the whitened lead field.
    lead_field_scaling: float = field(init=False)

    def __post_init__(self) -> None:
        # The lead field and the whitening filter are indexed by position, so a
        # mismatch in either would silently mix up channels or sources.
        n_sensors, n_sources = len(self.sensor), len(self.source) * self.dc
        if self.lead_field.shape != (n_sensors, n_sources):
            raise ValueError(f"lead_field of shape {self.lead_field.shape}; should be {(n_sensors, n_sources)} for {n_sensors} sensors and {len(self.source)} sources with {self.dc} orientation(s)")
        if self.noise_covariance.shape != (n_sensors, n_sensors):
            raise ValueError(f"noise covariance of shape {self.noise_covariance.shape}; should be {(n_sensors, n_sensors)} to match the {n_sensors} sensors of the lead field")
        self._prewhiten()

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {_forward_summary(self)}>'

    @classmethod
    def from_lead_field(cls, lead_field: NDVar, noise_covariance: NoiseArg) -> ForwardModel:
        """Construct from an Eelbrain lead-field :class:`~eelbrain.NDVar`.

        Parameters
        ----------
        lead_field
            Forward solution with ``sensor`` and ``source`` dimensions and an
            optional ``space`` dimension for free orientation.
        noise_covariance
            Noise covariance, aligned with the lead field's sensors: as
            :class:`mne.Covariance`, as :class:`eelbrain.NDVar` data from which a
            covariance is estimated, or as an already aligned covariance matrix
            (see :class:`~ncrf.NCRFEstimator`).
        """
        if lead_field.has_dim('space'):
            g = lead_field.get_data(dims=('sensor', 'source', 'space')).astype(np.float64)
            g = g.reshape(g.shape[0], -1)
            space = lead_field.get_dim('space')
        else:
            g = lead_field.get_data(dims=('sensor', 'source')).astype(np.float64)
            space = None
        sensor = lead_field.get_dim('sensor')
        return cls(g, _noise_covariance(noise_covariance, sensor).astype(np.float64), lead_field.get_dim('source'), sensor, space)

    @property
    def dc(self) -> int:
        """Number of orientation components per source."""
        return len(self.space) if self.space else 1

    @cached_property
    def mne_initializer(self) -> MNEInitializer:
        """MNE-style initializer for :attr:`~ncrf.ForwardModel.whitened_lead_field`."""
        return MNEInitializer(self.whitened_lead_field)

    def source_block(self, i: int) -> slice:
        """Column/row slice of source ``i``'s orientation components in stacked arrays."""
        dc = self.dc
        return slice(i * dc, (i + 1) * dc)

    def whiten(
            self,
            data: RegressionData,
            accept_whitening: bool = False,
    ) -> RegressionData:
        """Whiten ``data`` with :attr:`~ncrf.ForwardModel.whitening_filter`, after checking sensor alignment.

        Parameters
        ----------
        data
            Dataset to whiten; it has to have exactly this forward model's sensors,
            in the same order.
        accept_whitening
            Return an already-whitened dataset unchanged (see
            :meth:`RegressionData.whiten`).

        Raises
        ------
        ValueError
            If ``data`` has different sensors than the forward model, or is
            already whitened and ``accept_whitening`` is false.
        """
        _assert_sensors_equal(data.sensor_dim.names, self.sensor.names, 'data', 'forward model')
        return data.whiten(self.whitening_filter, accept_whitening=accept_whitening)

    def _prewhiten(self) -> None:
        """Compute whitened derived quantities from ``lead_field`` and ``noise_covariance``.

        Writes ``whitening_filter``, ``whitened_lead_field``, ``lead_field_scaling``,
        and ``whitened_noise_covariance``.  Neither ``lead_field`` nor
        ``noise_covariance`` is modified.
        """
        wf = _inv_sqrtm(self.noise_covariance)
        if (np.var(wf, axis=1) == 0).any():
            raise ValueError("Noise covariance data contains flat channels")
        self.whitening_filter = wf
        self.whitened_lead_field = np.dot(wf, self.lead_field)
        self.whitened_noise_covariance = wf.dot(self.noise_covariance).dot(wf.T)
        self.lead_field_scaling = linalg.norm(self.whitened_lead_field, 2)
        self.whitened_lead_field /= self.lead_field_scaling

    def __getstate__(self) -> dict[str, Any]:
        # Derived (whitened) quantities are recomputed by _prewhiten() on unpickling.
        return {
            'lead_field': self.lead_field,
            'noise_covariance': self.noise_covariance,
            'source': self.source,
            'sensor': self.sensor,
            'space': self.space,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._prewhiten()
