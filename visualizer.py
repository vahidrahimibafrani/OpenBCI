from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Lazy import typing for Matplotlib to avoid heavy import cost at module import.
try:  # pragma: no cover (typing-only convenience)
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    Axes = object  # type: ignore
    Figure = object  # type: ignore

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
            df: Optional[pd.DataFrame] = None
    ) -> None:
        self.df:Optional[pd.DataFrame] = df
        self.exg_prefix = "EXG Channel"
        self._default_candidates = [f"EXG Channel {i}" for i in range(8)]

    def __repr__(self) -> str:
        """Return a Developrt Friendly representation of the instance """
        if self.df in None:
            columns_repr = "<unloaded>"
            rows = 0
        else:
            columns = list(self.df.columns)
            rows = len(self.df)
            preview = ", ".join(columns[:5])
            if len(columns) > 5:
                preview += ", ..."
            columns_repr =f"[{preview}]"
        
        return f"{self.__class__.__name__}(rows={rows}, columns={columns_repr})"
    
    
    # --------- Public API ---------

    def stacked(
        self,
        columns: Optional[Sequence[str]] = None,
        *,
        figsize: Tuple[Union[int, float], Union[int, float]] = (12, 10),
        linewidth: float = 0.4,
        sharex: bool = True,
        title: Optional[str] = "Channels (stacked)",
        downsample: Optional[int] = None,
        rolling_window: Optional[int] = None,
        zscore: bool = False,
        detrend: bool = False,
        grid: bool = True,
    ) -> Figure:
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

        self._require_dataframe()
        cols = self._resolve_columns(columns)
        data = self._prepare_data(cols, downsample, rolling_window, zscore, detrend)

        nrows = data.shape[1]
        if nrows == 0:
            raise ValueError("No data avaiable for plotting")
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=figsize,
            sharex=sharex,
            squeeze=False,
        )

        for idx, column in enumerate(data.columns):
            ax = axes[idx, 0]
            ax.plot(
                data.index,
                data[column],
                linewidth=linewidth,
                label = column,
            )
            if grid:
                ax.grid(True, linestyle="--",linewidth=0.5,alpha=0.6)
            ax.legend(loc="upper left")
        if sharex:
            axes[-1,0].set_xlabel("Sample")
        
        if 0 < len(cols) <=4:
            axes[0,0].legend(loc="upper right")
        
        fig.tight_layout()
        if title:
            fig.suptitle(title)
            fig.subplots_adjust(top=0.92)

        return fig

    # --------- Helpers (internal) ---------

    def _resolve_columns(self, columns:Optional[Sequence[str]]) -> List[str]:
        """Valdiate/choose columns to plot"""
        
        self._require_dataframe()
        assert self.df is not None
        
        if columns :
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                raise KeyError(f"Columns Not in DataFrame: {missing}")
            
            return list(columns)
        
        exg_cols = [c for c in self.df.columns if c.startswith(self.exg_prefix)]
        if exg_cols:
            return exg_cols
        
        cand = [c for c in self._default_candidates if c in self.df.columns]
        if cand:
            return cand
        
        raise ValueError(f"Unable to infer channel columns from Dataframe")
        

    def _prepare_data(
            self, 
            cols: Sequence[str],
            downsample: Optional[int],
            rolling_window: Optional[int],
            zscore: bool,
            detrend:bool,
    ) -> pd.DataFrame:

        """
        Select columns and apply transforms without mutating `df` 
        """
        self._require_dataframe()
        assert self.df is not None
        
        data = self.df.loc[:, cols].copy()

        if detrend:
            data = self._detrend(data)

        if zscore:
            std = data.std(ddof = 0).replace(0, np.nan)
            data = (data - data.mean()) / std

        if rolling_window is not None and rolling_window < 1:
            raise ValueError("`rolling window` must be a positive integer")
        
        if rolling_window and rolling_window > 1:
            data = data.rolling(rolling_window, min_periods=1).mean()

        if downsample is not None and downsample < 1:
            raise ValueError("`downsample` must be a positive integer.")
        
        if downsample and downsample > 1:
            data = data.iloc[::downsample, :]

        if data.isna().all(axis=1).any():
            data = data.dropna(how="all")

        return data

    def _detrend(self, data:pd.DataFrame) -> pd.DataFrame:
        """remove a best-fit straight line from each column"""
        if data.empty:
            return data
        x = np.arange(len(data), dtype=float)
        detrended = np.empty_like(data.to_numpy(dtype=float))

        for idx, column in enumerate(data.columns):
            if self.exg_prefix in column:
                column_values = data.iloc[:, idx].to_numpy(dtype=float)
                mask = np.isfinite(column_values)
                if mask.sum() < 2:
                    detrended[:,idx] = column_values
                    continue
                x_valid = x[mask]
                y_valid = column_values[mask]

                A = np.column_stack((x_valid, np.ones_like(x_valid)))
                coeffs, *_ = np.linalg.lstsq(A, y_valid, rcond=None)
                trend = A @ coeffs
                
                result = column_values.copy()
                result[mask] = y_valid - trend
                detrended[: , idx] = result

        return pd.DataFrame(detrended, index=data.index, columns=data.columns)
    
    def _require_dataframe(self) ->None:
        """Ensure a DataFrame has been supplied"""
        if self.df is None:
            raise RuntimeError(
                "No DataFrame Provided. Initialize with DataFrame"
            )
if __name__ == "__main__":
    from datareader import DataReader
    dr = DataReader("./data/BrainFlow-RAW_2025-11-01_17-07-18_0.csv")
    data = dr.df
    vs = Visualizer(data)
    fig = vs.stacked(['EXG Channel 0','Digital Channel 0 (D11)',
                        'Digital Channel 1 (D12)', 'Digital Channel 2 (D13)',
                        'Digital Channel 3 (D17)', 'Digital Channel 4 (D18)'])
    fig.savefig('test.png')