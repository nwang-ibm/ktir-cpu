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

"""Unit tests for ktdp.construct_distributed_memory_view.

Covers the RFC §C.3 reference example (distributed-view-copy.mlir) with
data-correctness verification, plus a focused small-case test exercising
mixed-memory-space routing and differing per-partition strides.
"""

import numpy as np
import pytest

from ktir_cpu import KTIRInterpreter
from conftest import get_test_params


def _write_strided(mem, base_ptr: int, block: np.ndarray, strides):
    """Write ``block`` into *mem* so that element (r, c, ...) lands at
    byte offset ``base_ptr + sum(coord_d * strides[d]) * 2`` (f16).

    Uses a single dense buffer of length ``max_offset + 1`` so holes
    caused by non-contiguous strides are preserved as zeros.
    """
    assert block.dtype == np.float16
    ndim = block.ndim
    assert len(strides) == ndim
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in block.shape], indexing='ij'), axis=-1)
    coords = coords.reshape(-1, ndim)
    offsets = coords @ np.array(strides, dtype=np.int64)
    span = int(offsets.max()) + 1 if offsets.size else 1
    buf = np.zeros(span, dtype=np.float16)
    buf[offsets] = block.flatten()
    mem.write(base_ptr, buf)


# ---------------------------------------------------------------------------
# RFC example: distributed-view-copy.mlir (moved from test_spec_gaps.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path,func_name,entry", get_test_params("distributed_view_copy"))
def test_distributed_view_copy_rfc(path, func_name, entry):
    """construct_distributed_memory_view — RFC §C.3 example file.

    A is a 192×64 logical tensor distributed across three regions:
      A_HBM (96×64, HBM,         row-major strides [64, 1])  rows   0..95
      A_LX0 (32×64, LX, tile 0, col-packed strides [1, 64])  rows  96..127
      A_LX1 (64×64, LX, tile 0,  row-major strides [64, 1])  rows 128..191
    The kernel copies A into contiguous HBM output B, also 192×64.
    """
    interp = KTIRInterpreter()
    interp.load(path)
    _orig = interp._prepare_execution

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        lx0 = interp.memory.get_lx(0)

        # Global tensor: distinct value per (row, col) so mis-routing is
        # detectable.  Use row*64 + col so the f16 rounding does not
        # collapse neighbours (values 0..12287 fit f16 with occasional
        # rounding to the nearest-4 near the top, but row/col remain
        # recoverable from any block assertion below).
        full = np.arange(192 * 64, dtype=np.float16).reshape(192, 64)

        # A_HBM @ HBM 0 — rows 0..95, row-major
        hbm.write(0, full[0:96, :].flatten())

        # A_LX0 @ LX(0) 12288 — rows 96..127, strides [1, 64]
        _write_strided(lx0, 12288, full[96:128, :].copy(), strides=[1, 64])

        # A_LX1 @ LX(0) 20480 — rows 128..191, row-major
        lx0.write(20480, full[128:192, :].flatten())

        # B output @ HBM 24576 — zeroed
        hbm.write(24576, np.zeros(192 * 64, dtype=np.float16))

    interp._prepare_execution = _prepare_and_seed
    interp.execute_function(func_name)

    expected = np.arange(192 * 64, dtype=np.float16).reshape(192, 64)
    b = interp.memory.hbm.read(24576, 192 * 64, "f16").reshape(192, 64)
    np.testing.assert_array_equal(b, expected)


# ---------------------------------------------------------------------------
# Focused 4x4 unit test: 2 partitions, mixed memory spaces, differing strides
# ---------------------------------------------------------------------------

_SMALL_MLIR = """
#A0_set = affine_set<(d0, d1) : (d0 >= 0,   -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0)>
#A1_set = affine_set<(d0, d1) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>
#B_set  = affine_set<(d0, d1) : (d0 >= 0,   -d0 + 3 >= 0, d1 >= 0, -d1 + 3 >= 0)>
#identity = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @small_distributed_copy() {
    %c0 = arith.constant 0 : index
    %A0_addr = arith.constant 0     : index
    %A1_addr = arith.constant 4096  : index
    %B_addr  = arith.constant 8192  : index

    // A0: rows 0..1, HBM, row-major (strides [4, 1])
    %A0 = ktdp.construct_memory_view %A0_addr, sizes: [2, 4], strides: [4, 1] {
        coordinate_set = #A0_set,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<2x4xf16>

    // A1: rows 2..3, LX, column-packed (strides [1, 4])
    %A1 = ktdp.construct_memory_view %A1_addr, sizes: [2, 4], strides: [1, 4] {
        coordinate_set = #A1_set,
        memory_space = #ktdp.spyre_memory_space<LX>
    } : memref<2x4xf16>

    %A_global = ktdp.construct_distributed_memory_view
        (%A0, %A1 : memref<2x4xf16>, memref<2x4xf16>) : memref<4x4xf16>

    // Output B on HBM, row-major
    %B = ktdp.construct_memory_view %B_addr, sizes: [4, 4], strides: [4, 1] {
        coordinate_set = #B_set,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<4x4xf16>

    %A_at = ktdp.construct_access_tile %A_global[%c0, %c0] {
        access_tile_set = #B_set,
        access_tile_order = #identity
    } : memref<4x4xf16> -> !ktdp.access_tile<4x4xindex>

    %B_at = ktdp.construct_access_tile %B[%c0, %c0] {
        access_tile_set = #B_set,
        access_tile_order = #identity
    } : memref<4x4xf16> -> !ktdp.access_tile<4x4xindex>

    %data = ktdp.load %A_at : !ktdp.access_tile<4x4xindex> -> tensor<4x4xf16>
    ktdp.store %data, %B_at : tensor<4x4xf16>, !ktdp.access_tile<4x4xindex>
    return
  }
}
"""


def test_small_distributed_copy_mixed_spaces_and_strides():
    """2-partition distributed view: HBM row-major + LX column-packed.

    Verifies end-to-end that (1) each global coord routes to the correct
    partition, (2) HBM and LX partitions coexist in one view, and
    (3) differing stride patterns per partition are honored.
    """
    interp = KTIRInterpreter()
    interp.load(_SMALL_MLIR)
    _orig = interp._prepare_execution

    def _prepare_and_seed(grid_shape):
        _orig(grid_shape)
        hbm = interp.memory.hbm
        lx0 = interp.memory.get_lx(0)
        full = np.arange(16, dtype=np.float16).reshape(4, 4)
        # A0: rows 0..1 of full, row-major in HBM.
        hbm.write(0, full[0:2, :].flatten())
        # A1: rows 2..3 of full, strided [1, 4] in LX.
        _write_strided(lx0, 4096, full[2:4, :].copy(), strides=[1, 4])
        # B output zeroed.
        hbm.write(8192, np.zeros(16, dtype=np.float16))

    interp._prepare_execution = _prepare_and_seed
    interp.execute_function("small_distributed_copy")

    b = interp.memory.hbm.read(8192, 16, "f16").reshape(4, 4)
    expected = np.arange(16, dtype=np.float16).reshape(4, 4)
    np.testing.assert_array_equal(b, expected)
