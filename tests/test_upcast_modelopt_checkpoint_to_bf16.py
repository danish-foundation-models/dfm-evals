from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors")
safe_open = safetensors.safe_open
save_file = pytest.importorskip("safetensors.torch").save_file
if not hasattr(torch, "float8_e4m3fn"):
    pytest.skip("requires torch float8 support", allow_module_level=True)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "lumi" / "upcast_modelopt_checkpoint_to_bf16.py"
)

FP4_TO_NIBBLE = {
    0.0: 0x0,
    0.5: 0x1,
    1.0: 0x2,
    1.5: 0x3,
    2.0: 0x4,
    3.0: 0x5,
    4.0: 0x6,
    6.0: 0x7,
    -0.0: 0x0,
    -0.5: 0x9,
    -1.0: 0xA,
    -1.5: 0xB,
    -2.0: 0xC,
    -3.0: 0xD,
    -4.0: 0xE,
    -6.0: 0xF,
}


def pack_fp4_rows(values: torch.Tensor) -> torch.Tensor:
    rows, cols = values.shape
    assert cols % 2 == 0
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8)
    for row in range(rows):
        for col in range(0, cols, 2):
            low = FP4_TO_NIBBLE[float(values[row, col].item())]
            high = FP4_TO_NIBBLE[float(values[row, col + 1].item())]
            packed[row, col // 2] = low | (high << 4)
    return packed


def linear_to_swizzled_scale(
    linear: torch.Tensor,
    output_size: int,
    input_size: int,
    group_size: int,
) -> torch.Tensor:
    output_tiles = (output_size + 128 - 1) // 128
    packed_tile_width = group_size * 4
    input_tiles = (input_size + packed_tile_width - 1) // packed_tile_width
    padded = torch.zeros(
        (output_tiles * 128, input_tiles * packed_tile_width // group_size),
        dtype=linear.dtype,
    )
    padded[: linear.shape[0], : linear.shape[1]] = linear
    reshaped = padded.reshape(1, output_tiles, 4, 32, input_tiles, 4)
    return reshaped.permute(0, 1, 4, 3, 2, 5).reshape(
        output_tiles * 128,
        input_tiles * packed_tile_width // group_size,
    )


def test_upcast_modelopt_checkpoint_to_bf16(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    blobs_dir = tmp_path / "blobs"
    source_dir.mkdir()
    blobs_dir.mkdir()

    fp8_weight = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    fp8_weight_scale = torch.tensor(2.0, dtype=torch.float32)
    fp8_input_scale = torch.tensor(7.0, dtype=torch.float32)
    fp8_k_scale = torch.tensor(1.0, dtype=torch.float32)
    fp8_v_scale = torch.tensor(1.0, dtype=torch.float32)

    nvfp4_values = torch.tensor(
        [
            [
                0.5,
                -1.0,
                1.5,
                2.0,
                *([0.0] * 12),
                -0.5,
                1.0,
                -1.5,
                3.0,
                *([0.0] * 12),
                4.0,
                -6.0,
                0.5,
                -0.5,
                *([0.0] * 12),
                1.0,
                1.5,
                -2.0,
                0.5,
                *([0.0] * 12),
            ]
        ],
        dtype=torch.float32,
    )
    nvfp4_weight = pack_fp4_rows(nvfp4_values)
    nvfp4_weight_scale_linear = torch.tensor(
        [[2.0, 4.0, 6.0, 8.0]], dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    nvfp4_weight_scale = linear_to_swizzled_scale(
        nvfp4_weight_scale_linear,
        output_size=1,
        input_size=64,
        group_size=16,
    )
    nvfp4_weight_scale_2 = torch.tensor(2.0, dtype=torch.float32)
    nvfp4_input_scale = torch.tensor(5.0, dtype=torch.float32)

    bf16_weight = torch.tensor([[9.0, 10.0]], dtype=torch.bfloat16)

    tensors = {
        "fp8_layer.weight": fp8_weight,
        "fp8_layer.weight_scale": fp8_weight_scale,
        "fp8_layer.input_scale": fp8_input_scale,
        "fp8_layer.k_scale": fp8_k_scale,
        "fp8_layer.v_scale": fp8_v_scale,
        "nvfp4_layer.weight": nvfp4_weight,
        "nvfp4_layer.weight_scale": nvfp4_weight_scale,
        "nvfp4_layer.weight_scale_2": nvfp4_weight_scale_2,
        "nvfp4_layer.input_scale": nvfp4_input_scale,
        "plain.weight": bf16_weight,
    }
    save_file(tensors, str(source_dir / "model-00001-of-00001.safetensors"))

    (source_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    key: "model-00001-of-00001.safetensors" for key in tensors
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "nemotron_h",
                "dtype": "bfloat16",
                "quantization_config": {"producer": {"name": "modelopt"}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "hf_quant_config.json").write_text(
        json.dumps({"producer": {"name": "modelopt"}}) + "\n",
        encoding="utf-8",
    )
    (blobs_dir / "configuration_nemotron_h.py").write_text(
        "class NemotronHConfig:\n    pass\n",
        encoding="utf-8",
    )
    (blobs_dir / "generation_config.json").write_text(
        json.dumps({"temperature": 0.1}) + "\n",
        encoding="utf-8",
    )
    (source_dir / "configuration_nemotron_h.py").symlink_to(
        blobs_dir / "configuration_nemotron_h.py"
    )
    (source_dir / "generation_config.json").symlink_to(
        blobs_dir / "generation_config.json"
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source",
            str(source_dir),
            "--output",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert not (output_dir / "hf_quant_config.json").exists()
    assert (output_dir / "configuration_nemotron_h.py").exists()
    assert not (output_dir / "configuration_nemotron_h.py").is_symlink()
    assert (output_dir / "generation_config.json").exists()
    assert not (output_dir / "generation_config.json").is_symlink()

    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert "quantization_config" not in config
    assert config["dtype"] == "bfloat16"
    assert config["torch_dtype"] == "bfloat16"

    with safe_open(output_dir / "model-00001-of-00001.safetensors", framework="pt") as fh:
        keys = set(fh.keys())
        assert keys == {"fp8_layer.weight", "nvfp4_layer.weight", "plain.weight"}

        fp8_out = fh.get_tensor("fp8_layer.weight")
        nvfp4_out = fh.get_tensor("nvfp4_layer.weight")
        plain_out = fh.get_tensor("plain.weight")

    assert fp8_out.dtype == torch.bfloat16
    assert nvfp4_out.dtype == torch.bfloat16
    assert plain_out.dtype == torch.bfloat16

    expected_fp8 = torch.tensor([[2.0, -4.0], [1.0, 6.0]], dtype=torch.float32).to(
        torch.bfloat16
    )
    expected_nvfp4 = (
        nvfp4_values.reshape(1, 4, 16)
        * (nvfp4_weight_scale_linear.to(torch.float32) * 2.0).unsqueeze(-1)
    ).reshape(1, 64).to(torch.bfloat16)
    expected_plain = bf16_weight

    assert torch.equal(fp8_out, expected_fp8)
    assert torch.equal(nvfp4_out, expected_nvfp4)
    assert torch.equal(plain_out, expected_plain)

    output_index = json.loads(
        (output_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    assert set(output_index["weight_map"]) == {
        "fp8_layer.weight",
        "nvfp4_layer.weight",
        "plain.weight",
    }
