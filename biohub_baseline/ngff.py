"""Small dependency-light reader for the competition's Zarr v3 arrays."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


def _decompress(raw: bytes, codecs: list[dict[str, object]]) -> bytes:
    names = [codec.get("name") for codec in codecs]
    if "blosc" in names:
        try:
            import blosc2

            return blosc2.decompress(raw)
        except ImportError:
            try:
                import numcodecs.blosc

                return numcodecs.blosc.decompress(raw)
            except ImportError as error:
                raise ImportError("a blosc decoder is required for competition chunks") from error
    return gzip.decompress(raw) if "gzip" in names else raw


class NgffArray:
    def __init__(self, path: Path):
        self.path = path
        metadata = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
        if metadata.get("node_type") != "array":
            raise ValueError(f"{path} is not a Zarr v3 array")
        self.shape = tuple(int(value) for value in metadata["shape"])
        self.chunks = tuple(
            int(value)
            for value in metadata["chunk_grid"]["configuration"]["chunk_shape"]
        )
        endian = "<"
        for codec in metadata.get("codecs", []):
            if codec.get("name") == "bytes" and codec["configuration"].get("endian") == "big":
                endian = ">"
        self.dtype = np.dtype(endian + np.dtype(metadata["data_type"]).str[1:])
        self.codecs = metadata.get("codecs", [])
        self.fill_value = metadata.get("fill_value", 0)
        self.separator = (
            metadata.get("chunk_key_encoding", {})
            .get("configuration", {})
            .get("separator", "/")
        )

    def _read_chunk(self, indexes: tuple[int, ...]) -> np.ndarray | None:
        key = self.separator.join(["c", *(str(index) for index in indexes)])
        chunk_path = self.path / key
        if not chunk_path.exists():
            return None
        decoded = _decompress(chunk_path.read_bytes(), self.codecs)
        return np.frombuffer(decoded, dtype=self.dtype).reshape(self.chunks)

    def __getitem__(self, time_index: int) -> np.ndarray:
        if not 0 <= time_index < self.shape[0]:
            raise IndexError(time_index)
        spatial_shape = self.shape[1:]
        output = np.full(spatial_shape, self.fill_value, dtype=self.dtype)
        time_chunk, time_offset = divmod(time_index, self.chunks[0])
        chunk_counts = [
            (size + chunk_size - 1) // chunk_size
            for size, chunk_size in zip(spatial_shape, self.chunks[1:])
        ]
        for z_index in range(chunk_counts[0]):
            for y_index in range(chunk_counts[1]):
                for x_index in range(chunk_counts[2]):
                    chunk = self._read_chunk((time_chunk, z_index, y_index, x_index))
                    if chunk is None:
                        continue
                    starts = np.asarray((z_index, y_index, x_index)) * self.chunks[1:]
                    stops = np.minimum(starts + self.chunks[1:], spatial_shape)
                    slices = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))
                    source = tuple(slice(0, int(stop - start)) for start, stop in zip(starts, stops))
                    output[slices] = chunk[time_offset][source]
        return output


def open_ngff(path: Path) -> NgffArray:
    array_path = path / "0" if (path / "0/zarr.json").exists() else path
    return NgffArray(array_path)
