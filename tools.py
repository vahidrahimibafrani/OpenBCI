""" 
    This file contains usefull functions to work with the OpenBCI data
"""
import numpy as np 
import pandas as pd
import datetime 
from typing import List, Union, Tuple

DIGITAL_CHN = [
    'Digital Channel 0 (D11)',
    'Digital Channel 1 (D12)',
    'Digital Channel 2 (D13)',
    'Digital Channel 3 (D17)',
    'Digital Channel 4 (D18)',
    ]  

def _normalize_for_match(s: str) -> str:
    """
    Normalize a channel string for matching:
    - drop everything in parentheses, e.g. 'Digital Channel 0 (D11)' -> 'Digital Channel 0'
    - strip and lowercase
    """
    base = s.split('(')[0]
    return base.strip().lower()

def _find_matches(dist: List[str], src: Union[str, List[str]]) -> List[str]:
    """
    Find matching strings in `dist` for the given `src` (single string or list of strings).
    Parenthesized parts of `dist` entries are ignored for the comparison.

    Examples:
        dist = ['Digital Channel 0 (D11)', 'Digital Channel 1 (D12)']
        src  = '0'  -> matches ['Digital Channel 0 (D11)']
        src  = '1'  -> matches ['Digital Channel 1 (D12)']
    """
    if isinstance(src, str):
        src = [src]

    # Pre-normalize dist entries (but keep originals for return)
    normalized_dist = [(original, _normalize_for_match(original)) for original in dist]

    results: List[str] = []
    for m in src:
        # accept ints too, just in case
        if not isinstance(m, str):
            m = str(m)
        m_norm = m.strip().lower()
        if not m_norm:
            continue

        for original, norm in normalized_dist:
            # Only compare against the normalized version (no parentheses)
            if m_norm in norm or norm in m_norm:
                results.append(original)

    # Deduplicate while preserving order
    seen = set()
    deduped = [s for s in results if not (s in seen or seen.add(s))]
    return deduped
    
def marker_datection(df: pd.DataFrame, marker: List[int], channels: List[str]) -> np.ndarray:
    """
    Detection of indicated marker occurrences in selected digital channels.

    Assumptions:
    - Markers are present on the digital channels:
        'Digital Channel 0 (D11)',
        'Digital Channel 1 (D12)',
        'Digital Channel 2 (D13)',
        'Digital Channel 3 (D17)',
        'Digital Channel 4 (D18)'
    - `channels` can contain either:
        - the channel numbers as strings, e.g. ['0', '3'], or
        - (part of) the channel names, e.g. ['Digital Channel 0', 'Channel 3']
    - `marker[i]` is the value we are looking for in the channel corresponding to `channels[i]`.

    The function returns the DataFrame indices where **all** the given markers
    occur simultaneously on their corresponding channels.
    """

    if len(marker) != len(channels):
        raise ValueError(
            f"the length of the marker and channel inputs should be the same "
            f"{len(marker)} != {len(channels)}"
        )

    # Resolve each user-provided channel identifier to an exact DIGITAL_CHN name
    resolved_channels: List[str] = []
    for ch in channels:
        matches = _find_matches(DIGITAL_CHN, ch)
        if not matches:
            raise ValueError(
                f"Could not resolve channel identifier '{ch}' to a known digital channel."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Channel identifier '{ch}' is ambiguous. It matched: {matches}. "
                "Please provide a more specific channel identifier."
            )
        resolved_channels.append(matches[0])

    # Make sure those channels exist in the DataFrame
    missing = [c for c in resolved_channels if c not in df.columns]
    if missing:
        raise KeyError(
            f"The DataFrame does not contain the expected digital channel columns: {missing}"
        )

    # Build a boolean mask where all markers match on their respective channels
    mask = pd.Series(True, index=df.index, dtype=bool)
    for ch_name, value in zip(resolved_channels, marker):
        mask &= (df[ch_name] == value)

    # Return the indices where all conditions hold
    indices = df.index[mask].to_numpy()
    return indices

def chunk_consecutive(indices: np.ndarray) -> List[Tuple[int, int]]:
    """
    Group consecutive indices into segments.

    Parameters
    ----------
    indices : np.ndarray
        1D array of integer indices (e.g. output of marker_datection).

    Returns
    -------
    List[Tuple[int, int]]
        List of (start_index, end_index) pairs (inclusive) for each
        consecutive run. If `indices` is empty, returns an empty list.

    Example
    -------
    indices = np.array([1, 2, 3, 7, 8, 10])
    -> [(1, 3), (7, 8), (10, 10)]
    """
    if indices.size == 0:
        return []

    # Make sure they are sorted (marker_datection should already do this,
    # but this makes the function robust)
    indices_sorted = np.sort(indices.astype(int))

    segments: List[Tuple[int, int]] = []
    start = int(indices_sorted[0])
    prev = int(indices_sorted[0])

    for idx in indices_sorted[1:]:
        idx = int(idx)
        if idx == prev + 1:
            # still in the same segment
            prev = idx
        else:
            # close current segment and start a new one
            segments.append((start, prev))
            start = idx
            prev = idx

    # close the last segment
    segments.append((start, prev))

    return segments

def denoise_chunks(chunks:List[Tuple[int,int]], threshold:int=50) -> List[Tuple[int,int]]:
    """
    remove small chunks

    Parametes
    ---------
    chunk: List[Tuple[int, int]]
        List of (start_index, end_index) pairs (inclusive) for each
        consecutive run. If `indices` is empty, returns an empty list.
    threshold: int
        the length threshhold of the chunks
    
    Returns
    -------
    List[Tuple[int,int]]
        List of (start_index, end_index) pairs (inclusive) for each
        consecutive run with lenght biger than threshold.
    """
    result = []
    for ch in chunks:
        if ch[1] - ch[0] > threshold:
            result.append(ch)

    return result

def time_stamp_converter(timestamp:float) ->datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp)

if __name__ == "__main__":
    x = time_stamp_converter(1762004368.476541)
    print(x, type(x))