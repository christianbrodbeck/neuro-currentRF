"""Forward model and noise covariance with derived whitened quantities."""
from __future__ import annotations

from typing import Any

from eelbrain import NDVar, Sensor, SourceSpace, Space, VolumeSourceSpace
import numpy as np
from scipy import linalg

from ._data import RegressionData
from ._linalg import _inv_sqrtm
from ._repr import _forward_summary
from ._typing import FloatArray


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
        Sensor-space noise covariance, shape ``(n_sensors, n_sensors)``.
    source
        Source dimension of the forward model.
    sensor
        Sensor dimension of the forward model.
    space
        Orientation dimension (``None`` for fixed orientation).
    """

    def __init__(
            self,
            lead_field: FloatArray,
            noise_covariance: FloatArray,
            source: SourceSpace | VolumeSourceSpace,
            sensor: Sensor,
            space: Space | None,
    ) -> None:
        self.lead_field = lead_field
        self.noise_covariance = noise_covariance
        self.source = source
        self.sensor = sensor
        self.space = space
        self._prewhiten()

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {_forward_summary(self)}>'

    @classmethod
    def from_lead_field(cls, lead_field: NDVar, noise_covariance: FloatArray) -> ForwardModel:
        """Construct from an Eelbrain lead-field :class:`NDVar`."""
        if lead_field.has_dim('space'):
            g = lead_field.get_data(dims=('sensor', 'source', 'space')).astype(np.float64)
            g = g.reshape(g.shape[0], -1)
            space = lead_field.get_dim('space')
        else:
            g = lead_field.get_data(dims=('sensor', 'source')).astype(np.float64)
            space = None
        return cls(g, noise_covariance.astype(np.float64), lead_field.get_dim('source'), lead_field.get_dim('sensor'), space)

    @property
    def dc(self) -> int:
        """Number of orientation components per source."""
        return len(self.space) if self.space else 1

    def source_block(self, i: int) -> slice:
        """Column/row slice of source ``i``'s orientation components in stacked arrays."""
        dc = self.dc
        return slice(i * dc, (i + 1) * dc)

    def whiten(
            self,
            data: RegressionData,
            accept_whitening: bool = False,
    ) -> RegressionData:
        """Whiten ``data`` with :attr:`whitening_filter`, after checking sensor alignment.

        Parameters
        ----------
        data
            Dataset to whiten. Its channel names have to match those of
            :attr:`sensor`, in the same order, because the whitening filter and
            the lead field are indexed by channel position, not by name.
        accept_whitening
            Return an already-whitened dataset unchanged (see
            :meth:`RegressionData.whiten`).

        Raises
        ------
        ValueError
            If ``data`` has different sensors than the forward model, or is
            already whitened and ``accept_whitening`` is false.
        """
        if list(data.sensor_dim.names) != list(self.sensor.names):
            raise ValueError(f"data sensors do not match the forward model (data: {len(data.sensor_dim)} channels, forward model: {len(self.sensor)}); the whitening filter and the lead field are ordered by channel, so the forward model has to be built for exactly these sensors, e.g. lead_field.sub(sensor=data.sensor_dim)")
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
