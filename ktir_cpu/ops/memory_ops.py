# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Memory compute helpers.

Tile view construction, sub-tile access, and HBM/LX load/store
primitives used by dialect handlers in ``ktir_cpu.dialects``.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from ..affine import AffineMap, AffineSet
from ..dialects.ktdp_helpers import eval_subscript_expr
from ..dtypes import bytes_per_elem as _bytes_per_elem, to_np_dtype as _to_np_dtype
from ..ir_types import Tile, TileRef, DistributedTileRef
from ..grid import CoreContext
from ..memory import HBMSimulator


def partition_origin(coord_set: AffineSet, global_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Find the per-dim minimum global coordinate covered by *coord_set*.

    For an axis-aligned rectangular partition (the common case expressed
    by affine sets like ``d_i - C >= 0, -d_i + D >= 0``), this is the
    partition's origin in the global index space, so
    ``local_coord[i] = global_coord[i] - origin[i]``.

    Works for any convex set shape by enumerating points within
    *global_shape* and taking the per-dim min; callers should cache the
    result per distributed view.
    """
    points = coord_set.enumerate(global_shape)
    if not points:
        raise ValueError(
            f"Partition coordinate_set is empty over shape {global_shape}: "
            f"{coord_set.source!r}"
        )
    ndim = len(global_shape)
    return tuple(min(p[i] for p in points) for i in range(ndim))


class MemoryOps:
    """Tile memory helpers — view, access, load, store."""

    @staticmethod
    def tile_view(
        context: CoreContext,
        ptr: int,
        shape: Tuple[int, ...],
        strides: List[int],
        memory_space: str,
        dtype: str = "f16",
        coordinate_set: Optional[str] = None,
    ) -> TileRef:
        """Create a memory layout descriptor (TileRef).

        Builds a tile reference describing a contiguous region in HBM or LX.

        Args:
            context: Core execution context
            ptr: Base pointer
            shape: Tile shape
            strides: Memory strides
            memory_space: "HBM" or "LX"
            dtype: Data type
            coordinate_set: Verbatim affine_set string, no evaluation

        Returns:
            TileRef describing the memory layout
        """
        return TileRef(
            base_ptr=ptr,
            shape=shape,
            strides=strides,
            memory_space=memory_space,
            dtype=dtype,
            coordinate_set=coordinate_set,
        )

    @staticmethod
    def tile_access(
        context: CoreContext,
        parent_ref: TileRef,
        indices: List[int],
        access_shape: Tuple[int, ...],
        base_map: AffineMap,
    ) -> TileRef:
        """Extract a sub-tile from a parent tile reference.

        Evaluates *base_map* with *indices* to obtain the base coordinates
        in the parent memref, then computes a byte offset using the parent
        strides.  The resulting base_ptr is always within the same allocation
        as parent_ref.base_ptr — this invariant is relied upon by load/store.

        Args:
            context: Core execution context
            parent_ref: Parent tile reference (memref)
            indices: Access indices (one per base_map input dim)
            access_shape: Shape of the accessed sub-tile
            base_map: AffineMap mapping indices → base coordinates

        Returns:
            TileRef for the sub-tile
        """
        base_coords = base_map.eval(indices)
        bpe = _bytes_per_elem(parent_ref.dtype)
        offset = sum(coord * stride for coord, stride in zip(base_coords, parent_ref.strides))
        new_ptr = parent_ref.base_ptr + offset * bpe

        return TileRef(
            base_ptr=new_ptr,
            shape=access_shape,
            strides=parent_ref.strides,
            memory_space=parent_ref.memory_space,
            dtype=parent_ref.dtype
        )

    @staticmethod
    def _is_contiguous(shape: Tuple[int, ...], strides: Tuple[int, ...]) -> bool:
        """Check if a shape/strides pair describes contiguous (row-major) memory."""
        expected_stride = 1
        for dim, stride in zip(reversed(shape), reversed(strides)):
            if stride != expected_stride:
                return False
            expected_stride *= dim
        return True

    @staticmethod
    def _write_to_lx(context: CoreContext, data: np.ndarray):
        """Write data into the core-local LX scratchpad.

        Advances ``next_ptr`` so subsequent writes don't collide.
        LX capacity accounting is handled by ``CoreContext.track_lx()``
        in ``_execute_operation`` — we only reserve address space here.
        All loaded Tiles always land in LX regardless of source memory space.
        """
        size = data.nbytes
        lx_ptr = context.lx.next_ptr
        context.lx.next_ptr += size
        context.lx.next_ptr = (context.lx.next_ptr + HBMSimulator.STICK_BYTES - 1) & ~(HBMSimulator.STICK_BYTES - 1)
        context.lx.write(lx_ptr, data)

    @staticmethod
    def _flat_memory_offsets(
        base_ptr: int,
        shape: Tuple[int, ...],
        strides: List[int],
        dtype: str,
        coords: Optional[List[Tuple[int, ...]]] = None,
    ) -> Tuple[List[int], int]:
        """Linearize N-d coordinates to flat element offsets and count unique HBM sticks.

        O(n) time, O(unique_sticks) memory for the stick set. For very large
        coord sizes a more efficient implementation may be needed.

        Returns:
            (offsets, unique_sticks) — flat element offsets and number of
            distinct HBM sticks touched.
        """
        offsets = []
        sticks = set()
        bpe = _bytes_per_elem(dtype)
        for coord in (coords if coords is not None else np.ndindex(*shape)):
            o = sum(c * s for c, s in zip(coord, strides))
            offsets.append(o)
            sticks.add((base_ptr + o * bpe) // HBMSimulator.STICK_BYTES)
        return offsets, len(sticks)

    @staticmethod
    def load(
        context: CoreContext,
        tile_ref: TileRef,
        coords: Optional[List[Tuple[int, ...]]] = None,
        result_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tile:
        """Load data from HBM or LX into LX and return a Tile.

        All loaded Tiles always land in LX regardless of source memory space:
        - HBM source → DMA read from HBM, write into LX scratchpad.
        - LX source  → logical copy within LX (no physical movement).

        When *coords* is given (coordinate-set path), gathers only the
        elements at those local coordinates and reshapes to *result_shape*.
        When *coords* is None, loads the full tile described by tile_ref
        (contiguous or strided).

        A single ``mem.read`` covers the entire element footprint; no
        per-element dict scans occur.

        Example — loading column 2 of a 4×4 f16 matrix (strided, coords=None)::

            # Parent 4×4 allocation at base_ptr=0x1000, values 0..15
            # tile_ref for column 2: base_ptr=0x1004, shape=(4,), strides=[4]
            #   flat offsets: [0*4, 1*4, 2*4, 3*4] = [0, 4, 8, 12]
            #   span = 13  (max offset + 1)
            #   mem.read(0x1004, 13) -> [2,3,4,5,6,7,8,9,10,11,12,13,14]
            #   gathered = flat[[0,4,8,12]] = [2, 6, 10, 14]  ✓

        Example — upper-triangular load from a 4×4 tile (coords provided)::

            # tile_ref: base_ptr=0x1000, shape=(4,4), strides=[4,1]
            # coords = [(0,0),(0,1),...,(3,3)]  — 10 upper-tri tuples
            #   flat offsets = [0*4+0, 0*4+1, ..., 3*4+3] = [0,1,2,3,5,6,7,10,11,15]
            #   span = 16
            #   mem.read(0x1000, 16) -> flat 0..15
            #   gathered = flat[[0,1,2,3,5,6,7,10,11,15]] = [0,1,2,3,5,6,7,10,11,15]

        Args:
            context: Core execution context
            tile_ref: Tile reference (memref) describing source
            coords: Optional list of local coordinate tuples to gather.
                    Each tuple is 0-based within tile_ref.shape.
            result_shape: Output shape when coords is given; defaults to
                          tile_ref.shape when coords is None.

        Returns:
            Tile value (tensor) loaded into LX
        """
        mem = context.hbm if tile_ref.memory_space == "HBM" else context.lx

        # Fast path: contiguous tile, no coord filtering — single dict-key read.
        if coords is None and MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides):
            n = int(np.prod(tile_ref.shape))
            data = mem.read(tile_ref.base_ptr, n, tile_ref.dtype).reshape(tile_ref.shape)
            MemoryOps._write_to_lx(context, data)
            bpe = _bytes_per_elem(tile_ref.dtype)
            end = tile_ref.base_ptr + n * bpe
            unique_sticks = (
                (end + HBMSimulator.STICK_BYTES - 1) // HBMSimulator.STICK_BYTES
                - tile_ref.base_ptr // HBMSimulator.STICK_BYTES
            )
            return Tile(data, tile_ref.dtype, tile_ref.shape, unique_sticks)

        # Strided or coord-set path: linearize coords, single read, numpy fancy-index.
        offsets, unique_sticks = MemoryOps._flat_memory_offsets(
            tile_ref.base_ptr, tile_ref.shape, tile_ref.strides, tile_ref.dtype, coords
        )
        span = max(offsets) + 1 if offsets else 1
        flat = mem.read(tile_ref.base_ptr, span, tile_ref.dtype)

        gathered = flat[offsets]
        out_shape = result_shape if result_shape is not None else tile_ref.shape
        data = gathered.reshape(out_shape)

        MemoryOps._write_to_lx(context, data)
        return Tile(data, tile_ref.dtype, out_shape, unique_sticks)

    @staticmethod
    def store(
        context: CoreContext,
        tile: Tile,
        tile_ref: TileRef,
        coords: Optional[List[Tuple[int, ...]]] = None,
    ):
        """Store tile data to HBM or LX.

        - HBM target → DMA write from LX to HBM.
        - LX target  → write directly to LX.

        When *coords* is given (coordinate-set path), scatters tile elements
        to those local coordinates via a read-modify-write on the allocation.
        When *coords* is None, stores the full tile (contiguous or strided).

        A single ``mem.read`` + ``mem.write`` covers the entire footprint;
        no per-element dict scans occur.

        Args:
            context: Core execution context
            tile: Tile value (tensor data) to store
            tile_ref: Tile reference (memref) describing destination
            coords: Optional list of local coordinate tuples to scatter into.
        """
        mem = context.hbm if tile_ref.memory_space == "HBM" else context.lx

        # Fast path: contiguous tile, no coord filtering — single dict-key write.
        if coords is None and MemoryOps._is_contiguous(tile_ref.shape, tile_ref.strides):
            mem.write(tile_ref.base_ptr, tile.data.flatten())
            return

        # Strided or coord-set path: read-modify-write via scatter offsets.
        offsets, _ = MemoryOps._flat_memory_offsets(
            tile_ref.base_ptr, tile_ref.shape, tile_ref.strides, tile_ref.dtype, coords
        )
        span = max(offsets) + 1 if offsets else 1
        flat = mem.read(tile_ref.base_ptr, span, tile_ref.dtype)
        flat[offsets] = tile.data.flatten()
        mem.write(tile_ref.base_ptr, flat)

    @staticmethod
    def _group_by_partition(
        dist_ref: DistributedTileRef,
        coords: List[Tuple[int, ...]],
    ) -> Dict[int, Tuple[List[int], List[Tuple[int, ...]]]]:
        """Route each global coord to a partition and translate to local.

        Returns ``{partition_idx: (orig_positions, local_coords)}``.
        ``orig_positions[k]`` is the index into *coords* that produced
        ``local_coords[k]`` — used to scatter per-partition results back
        into a single flat output.

        Per-partition origins are computed lazily (and only once) from
        each partition's ``coordinate_set`` via :func:`partition_origin`.
        """
        ndim = len(dist_ref.shape)
        origins: Dict[int, Tuple[int, ...]] = {}
        groups: Dict[int, Tuple[List[int], List[Tuple[int, ...]]]] = {}
        for pos, coord in enumerate(coords):
            part_idx, part = dist_ref.find_partition(coord)
            if part_idx not in origins:
                origins[part_idx] = partition_origin(part.coordinate_set, dist_ref.shape)
            origin = origins[part_idx]
            local = tuple(int(coord[d] - origin[d]) for d in range(ndim))
            pos_list, local_list = groups.setdefault(part_idx, ([], []))
            pos_list.append(pos)
            local_list.append(local)
        return groups

    @staticmethod
    def load_distributed(
        context: CoreContext,
        dist_ref: DistributedTileRef,
        coords: List[Tuple[int, ...]],
        result_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tile:
        """Gather elements from a distributed view, route per partition.

        For each global coord, pick the partition whose ``coordinate_set``
        contains it, translate to that partition's local coord, issue one
        batched read per partition group, and scatter results back into a
        single flat output indexed by the coord's original position.
        Writes the result to LX and returns a :class:`Tile` so the caller
        sees the same contract as :meth:`load`.
        """
        out_shape = result_shape if result_shape is not None else tuple(dist_ref.shape)
        n_total = len(coords)
        np_dtype_output = np.zeros(n_total, dtype=_to_np_dtype(dist_ref.dtype))

        groups = MemoryOps._group_by_partition(dist_ref, coords)

        total_unique_sticks = 0
        for part_idx, (orig_positions, local_coords) in groups.items():
            part = dist_ref.partitions[part_idx]
            mem = context.hbm if part.memory_space == "HBM" else context.lx
            offsets, unique_sticks = MemoryOps._flat_memory_offsets(
                part.base_ptr, part.shape, part.strides, part.dtype, local_coords
            )
            span = max(offsets) + 1 if offsets else 1
            flat = mem.read(part.base_ptr, span, part.dtype)
            gathered = flat[offsets]
            # Scatter this partition's values back into their original positions.
            np_dtype_output[orig_positions] = gathered
            # TODO: stick counting across mixed memory spaces is approximate;
            # HBM partitions contribute real DMA traffic, LX partitions don't.
            total_unique_sticks += unique_sticks

        data = np_dtype_output.reshape(out_shape)
        MemoryOps._write_to_lx(context, data)
        return Tile(data, dist_ref.dtype, out_shape, total_unique_sticks)

    @staticmethod
    def store_distributed(
        context: CoreContext,
        tile: Tile,
        dist_ref: DistributedTileRef,
        coords: List[Tuple[int, ...]],
    ) -> None:
        """Scatter a tile across a distributed view, route per partition.

        Symmetric to :meth:`load_distributed`: group scatter positions by
        partition, then do one read-modify-write per partition using the
        existing ``_flat_memory_offsets`` machinery.
        """
        groups = MemoryOps._group_by_partition(dist_ref, coords)
        flat_values = tile.data.flatten()

        for part_idx, (orig_positions, local_coords) in groups.items():
            part = dist_ref.partitions[part_idx]
            mem = context.hbm if part.memory_space == "HBM" else context.lx
            offsets, _ = MemoryOps._flat_memory_offsets(
                part.base_ptr, part.shape, part.strides, part.dtype, local_coords
            )
            span = max(offsets) + 1 if offsets else 1
            flat = mem.read(part.base_ptr, span, part.dtype)
            flat[offsets] = flat_values[orig_positions]
            mem.write(part.base_ptr, flat)

    @staticmethod
    def indirect_load(
        context: CoreContext,
        iat: "IndirectAccessTile",
        result_shape: Optional[Tuple[int, ...]] = None,
    ) -> Tile:
        """Load data using an indirect access tile (gather pattern).

        Enumerates the variable space, resolves each coordinate tuple
        (direct dims use the variable value, indirect dims look up the
        index in an index memref), then delegates to :meth:`load`.
        """
        vss = iat.variables_space_set
        vso = iat.variables_space_order

        # Enumerate all points in the variable space
        points = vss.enumerate(iat.shape)
        if vso is not None:
            points = [vso.eval(pt) for pt in points]

        # For each point, resolve the actual coordinates in the parent memref
        coords = []
        for pt in points:
            coord = []
            for sub in iat.dim_subscripts:
                if sub["kind"] == "indirect":
                    # Look up the index from the index memref
                    idx_view = iat.index_views[sub["index_view_idx"]]
                    # Compute address into the index tensor
                    idx_coords = tuple(
                        eval_subscript_expr(e, pt) for e in sub["idx_exprs"]
                    )
                    mem = context.hbm if idx_view.memory_space == "HBM" else context.lx
                    offset = sum(c * s for c, s in zip(idx_coords, idx_view.strides))
                    addr = idx_view.base_ptr + offset * _bytes_per_elem(idx_view.dtype)
                    raw = mem.read(addr, 1, idx_view.dtype)
                    coord.append(int(raw[0]))
                elif sub["kind"] == "direct":
                    coord.append(pt[sub["var_index"]])
                elif sub["kind"] == "direct_expr":
                    coord.append(eval_subscript_expr(sub["subscript"], pt))
            coords.append(tuple(coord))

        out_shape = result_shape if result_shape is not None else iat.shape
        return MemoryOps.load(context, iat.parent_ref, coords=coords, result_shape=out_shape)
