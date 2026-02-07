import pytest
import numpy as np
import pandas as pd

from gnssvod.position.interpolation import sp3_interp_fast

@pytest.mark.parametrize(
    'start_time, end_time',
    [
        (pd.to_datetime('2020-01-01 00:00:00'),pd.to_datetime('2020-01-01 23:00:00')),
        (pd.to_datetime('2022-01-01 00:00:00'),pd.to_datetime('2022-01-01 23:00:00')),
        (pd.to_datetime('2024-01-01 00:00:00'),pd.to_datetime('2024-01-01 23:00:00')),
        (pd.to_datetime('2025-01-01 00:00:00'),pd.to_datetime('2025-01-01 23:00:00')),
        (pd.to_datetime('2026-01-01 00:00:00'),pd.to_datetime('2026-01-01 23:00:00'))
    ]
)
def test_sp3_interp_fast(start_time: pd.Timestamp, end_time: pd.Timestamp, tmp_path) -> None:
    interval = 30
    orbit = sp3_interp_fast(start_time,end_time,interval=interval,aux_path=tmp_path)
    # output is a dataframe
    assert isinstance(orbit, pd.DataFrame)
    # output contains complete records
    assert (len(orbit.dropna())>100)
    # output contains all required timestamps
    expected_timestamps = pd.date_range(start_time,end_time,freq=f"{interval}s")
    missing_values = np.setdiff1d(expected_timestamps,orbit.index.get_level_values('Epoch').unique())
    assert len(missing_values)==0
    # output contains SVs from all four constellations
    SVlist = np.array(orbit.index.get_level_values('SV').unique())
    SVconst = SVlist.astype('U1')
    missing_constellation = np.setdiff1d(['G','R','E','C'],SVconst)
    assert len(missing_constellation)==0

