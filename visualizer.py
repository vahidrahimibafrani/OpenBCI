from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union, overload

import numpy as np
import pandas as pd

# Lazy import typing for Matplotlib to avoid heavy import cost at module import.
try:  # pragma: no cover (typing-only convenience)
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    Axes = object  # type: ignore
    Figure = object  # type: ignore

Number = Union[int, float, np.number]


class Visualizer:
    """
    A typed, reusable plotting helper for multichannel timeseries (e.g., EXG/EMG/EEG)

    Assumptions 
    ---------------
        - `df` contains timeseries data in columns. the index can be timeseries or numeric.
        - You can pass a list of columns to plot; otherwisee we will auto-detect typical EXG names
        - provides stacked and grid plots with optional downsampling, smoothing, and per-channel z-score
    Notes  
    ---------------
        - This class does not mutate `df`. Transformations are applied to a working copy
        - for very large datasets, consider setting the `downsample` to speed up rendering  
    """

    def __init__(
            self,
            df: pd.DataFrame = None
    ) -> None:
        self.df = df
        self.exg_prefix = "EXG Channel"
        self._defualt_candidates = [f"EXG Channel {i}" for i in range(8)]

    def __repr__(self) -> str:
        pass

    # --------- Public API ---------

    def stacked(
            self,
            columns: Optional[Sequence[str]],
            *,
            figsize: Tuple[Union[int, float], Union[int, float]] = (12, 10),
            linewidth: float = 0.8,
            sharex: bool = True,
            title: Optional[str] = "Channels (stacked)",
            downsample: Optional[int] = None,
            rolling_window: Optional[int] = None,
            zscore: bool = False,
            detrend: bool = False,
            grid: bool = True,
    ) -> "Figure":
        """
        Plot channels as vertically stacked subplots.

        Parameters
        ----------
        columns : list-like[str] | None
            Channels to plot. If None, auto-detect columns that start with `exg_prefix`
            or fall back to `_default_candidates` intersection with df.columns.
        figsize : (w, h)
        linewidth : float
        sharex : bool
        title : str | None
        downsample : int | None
            Plot every Nth sample if provided (>=2).
        rolling_window : int | None
            Apply simple moving average of given window (samples).
        zscore : bool
            Standardize each channel to mean 0, std 1 (computed per column).
        detrend : bool
            Remove linear trend per column.
        grid : bool
            Show grid on each subplot.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        cols = self._resolve_columns(columns)
        data = self._prepare_data(cols, downsample, rolling_window, zscore, detrend)

        nrows = data.shape[1]

    # --------- Helpers (internal) ---------

    def _resolve_columns(self, columns:Optional[Sequence[str]]) -> List[str]:
        """Valdiate/choose columns to plot"""
        if columns is not None and len(columns) > 0:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                raise KeyError(f"Columns Not in DataFrame: {missing}")
            
            return columns
        
        exg_cols = [c for c in self.df.columns is c.starswith(self.exg_prefix)]
        
        if exg_cols:
            return exg_cols
        
        cand = [c for c in self._defualt_candidates if c in self.df.columns]
        
        if cand:
            return cand
        
        raise ValueError(f"Unable to parse columns")
        

    def _prepare_data(
            self, 
            cols: Sequence[str],
            downsample: Optional[int],
            rolling_window: Optional[int],
            zscore: bool,
            detrend:bool,
    ) -> pd.DataFrame

        """
        Select columns and apply transforms without mutating `df` 
        """
        data = self.df.loc[:, cols].copy()

        if detrend:
            data = self._deternd(data)

        if zscore:
            data = (data - data.mean()) / data.std(ddof = 0).replace(0, np.nan)

        if  rolling_window and rolling_window > 1:
            data = data.rolling(rolling_window, min_periods=1).mean()

        if downsample and downsample > 1:
            data = data.iloc[::downsample, :]

        if data.isna().all(axis=1).any():
            data = data.dropna(how='all')

        return data

    def _detrend()     
if __name__ == "__main__":
    vs = Visualizer()
