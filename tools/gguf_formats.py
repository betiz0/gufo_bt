#!/usr/bin/env python3
"""Report a GGUF artifact's identity, tensor formats and weight-class histogram.

Reads only the header of each shard: metadata key/value pairs and the tensor
infos (name, dimensions, storage type, offset). Prints the `general.*` identity
keys, the tensor count, the per-format histogram (tensors, elements, encoded
bytes) and the per-weight-class histogram the Flash-Next loader's format table
is built from, plus the SHA-256 of every shard when asked.

Usage:
  tools/gguf_formats.py MODEL.gguf [--sha256] [--json OUT.json]

A sharded target is given by its first shard; the remaining shards are derived
from the `split.count` metadata and the `-NNNNN-of-MMMMM` file-name suffix.
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys

MAGIC = 0x46554747

FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ2_S", 256, 82),
    19: ("IQ3_XXS", 256, 98),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ1_S", 256, 50),
    23: ("IQ4_XS", 256, 136),
    30: ("BF16", 1, 2),
}

# Weight classes of the Flash-Next format table
# (src/models/qwen38_flash_next/weight_format.hpp), keyed by tensor name.
CLASSES = [
    ("dense", ("token_embd.weight", "output.weight", "attn_qkv.weight",
               "attn_gate.weight", "attn_q.weight", "attn_k.weight",
               "attn_v.weight", "attn_output.weight", "ssm_out.weight",
               "indexer.q_proj.weight", "indexer.k_proj.weight",
               "ple_key.weight", "ple_value.weight", "ffn_gate_shexp.weight",
               "ffn_up_shexp.weight", "ffn_down_shexp.weight",
               "nextn.eh_proj.weight")),
    ("routed_gate_up", ("ffn_gate_exps.weight", "ffn_up_exps.weight")),
    ("routed_down", ("ffn_down_exps.weight",)),
    ("mixer_down_up", ("hc_attn_down.weight", "hc_attn_up.weight",
                       "hc_ffn_down.weight", "hc_ffn_up.weight",
                       "output_hc_down.weight", "output_hc_up.weight",
                       "hc_head_down.weight", "hc_head_up.weight")),
    ("mixer_inject", ("hc_attn_inject.weight", "hc_ffn_inject.weight",
                      "hc_head_inject.weight")),
    ("ssm_gate", ("ssm_alpha.weight", "ssm_beta.weight")),
    ("ple_table", ("per_layer_token_embd.weight",)),
    ("ple_conv1d", ("ple_conv1d.weight",)),
]


def class_of(name):
    match = re.match(r"^(?:blk\.\d+\.|nextn\.)(.*)$", name)
    suffix = match.group(1) if match else name
    suffix = suffix.replace("hc_head_", "hc_head_")
    for label, names in CLASSES:
        if suffix in names:
            return label
    return "f32_only"


def read_exact(f, count):
    data = f.read(count)
    if len(data) != count:
        raise SystemExit("truncated GGUF header")
    return data


def read_u32(f):
    return struct.unpack("<I", read_exact(f, 4))[0]


def read_u64(f):
    return struct.unpack("<Q", read_exact(f, 8))[0]


def read_string(f):
    length = read_u64(f)
    return read_exact(f, length).decode("utf-8", "replace")


def skip_value(f, value_type):
    if value_type == 8:
        read_string(f)
    elif value_type == 9:
        item_type = read_u32(f)
        count = read_u64(f)
        for _ in range(count):
            skip_value(f, item_type)
    elif value_type in FIXED_SIZES:
        read_exact(f, FIXED_SIZES[value_type])
    else:
        raise SystemExit(f"unknown GGUF value type {value_type}")


def read_value(f, value_type):
    if value_type == 8:
        return read_string(f)
    if value_type == 9:
        item_type = read_u32(f)
        count = read_u64(f)
        values = [read_value(f, item_type) for _ in range(count)]
        return values
    raw = read_exact(f, FIXED_SIZES[value_type])
    if value_type == 0:
        return raw[0]
    if value_type == 1:
        return struct.unpack("<b", raw)[0]
    if value_type == 2:
        return struct.unpack("<H", raw)[0]
    if value_type == 3:
        return struct.unpack("<h", raw)[0]
    if value_type == 4:
        return struct.unpack("<I", raw)[0]
    if value_type == 5:
        return struct.unpack("<i", raw)[0]
    if value_type == 6:
        return struct.unpack("<f", raw)[0]
    if value_type == 7:
        return raw[0] != 0
    if value_type == 10:
        return struct.unpack("<Q", raw)[0]
    if value_type == 11:
        return struct.unpack("<q", raw)[0]
    return struct.unpack("<d", raw)[0]


def read_shard(path):
    """Returns (metadata dict, tensor list) for one shard."""
    metadata = {}
    tensors = []
    with open(path, "rb") as f:
        if read_u32(f) != MAGIC:
            raise SystemExit(f"{path}: not a GGUF file")
        version = read_u32(f)
        tensor_count = read_u64(f)
        metadata_count = read_u64(f)
        for _ in range(metadata_count):
            key = read_string(f)
            value_type = read_u32(f)
            if key == "tokenizer.chat_template":
                skip_value(f, value_type)
                continue
            metadata[key] = read_value(f, value_type)
        for _ in range(tensor_count):
            name = read_string(f)
            n_dims = read_u32(f)
            dims = [read_u64(f) for _ in range(n_dims)]
            tensor_type = read_u32(f)
            offset = read_u64(f)
            tensors.append({"name": name, "dims": dims, "type": tensor_type,
                            "offset": offset})
        if version < 2:
            raise SystemExit(f"{path}: unsupported GGUF version {version}")
    return metadata, tensors


def shard_paths(first):
    stem = os.path.basename(first)
    match = re.match(r"^(.*)-(\d+)-of-(\d+)\.gguf$", stem)
    if match is None:
        return [first]
    prefix, _, total = match.groups()
    directory = os.path.dirname(os.path.abspath(first))
    return [os.path.join(directory, f"{prefix}-{index + 1:05d}-of-{total}.gguf")
            for index in range(int(total))]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="first shard of the target")
    parser.add_argument("--sha256", action="store_true",
                        help="also print each shard's SHA-256")
    parser.add_argument("--json", metavar="PATH",
                        help="write the report as JSON")
    args = parser.parse_args()

    metadata = {}
    tensors = []
    shards = []
    for path in shard_paths(args.model):
        if not os.path.exists(path):
            continue
        shard_meta, shard_tensors = read_shard(path)
        metadata.update(shard_meta)
        tensors.extend(shard_tensors)
        entry = {"path": os.path.basename(path),
                 "bytes": os.path.getsize(path)}
        if args.sha256:
            entry["sha256"] = sha256(path)
        shards.append(entry)
    if not tensors:
        raise SystemExit("no tensors found; is the first shard correct?")

    print(f"file {args.model}")
    for key in sorted(metadata):
        if key.startswith("general.") or key in ("split.count",):
            value = metadata[key]
            if isinstance(value, list):
                value = f"[{len(value)} values]"
            print(f"  {key} = {value}")
    print(f"  shards = {len(shards)}")
    for entry in shards:
        line = f"    {entry['path']} {entry['bytes']} bytes"
        if "sha256" in entry:
            line += f" sha256={entry['sha256']}"
        print(line)
    print(f"tensors {len(tensors)}")

    histograms = {}
    classes = {}
    for tensor in tensors:
        label, block, size = GGML_TYPES.get(
            tensor["type"], (f"type{tensor['type']}", 1, 0))
        elements = 1
        for dim in tensor["dims"]:
            elements *= dim
        if block > 1 and tensor["dims"][0] % block:
            raise SystemExit(
                f"{tensor['name']}: {label} does not divide dim0 "
                f"{tensor['dims'][0]}")
        encoded = elements // block * size
        entry = histograms.setdefault(label, {"tensors": 0, "elements": 0,
                                              "bytes": 0})
        entry["tensors"] += 1
        entry["elements"] += elements
        entry["bytes"] += encoded
        name_class = class_of(tensor["name"])
        entry = classes.setdefault(name_class, {"tensors": 0, "bytes": 0,
                                                "formats": {}})
        entry["tensors"] += 1
        entry["bytes"] += encoded
        entry["formats"][label] = entry["formats"].get(label, 0) + 1

    print("format histogram")
    for label in sorted(histograms, key=lambda k: -histograms[k]["bytes"]):
        entry = histograms[label]
        print(f"  {label:9s} tensors {entry['tensors']:5d} "
              f"elements {entry['elements']:>14d} bytes {entry['bytes']:>14d}")
    print("weight-class histogram")
    for label in sorted(classes):
        entry = classes[label]
        formats = ", ".join(f"{k}x{v}" for k, v in sorted(entry["formats"].items()))
        print(f"  {label:14s} tensors {entry['tensors']:5d} "
              f"bytes {entry['bytes']:>14d} [{formats}]")

    if args.json:
        with open(args.json, "w") as out:
            json.dump({"file": args.model, "shards": shards,
                       "metadata": {k: v for k, v in metadata.items()
                                    if k.startswith("general.")},
                       "tensor_count": len(tensors),
                       "formats": histograms, "classes": classes}, out, indent=1)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
