#!/usr/bin/env python3
import json
import argparse

def load_rms_from_onnx(onnx_path: str, key: str = "dreamwaq.rms"):
    import onnx

    model = onnx.load(onnx_path)
    for prop in model.metadata_props:
        if prop.key == key:
            if not prop.value:
                raise ValueError(f"metadata key '{key}' exists but value is empty")
            data = json.loads(prop.value)
            if not isinstance(data, dict):
                raise TypeError(f"metadata key '{key}' is not a dict json")
            return data
    raise KeyError(f"metadata key '{key}' not found")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", help="path to policy.onnx")
    parser.add_argument("--key", default="dreamwaq.rms", help="metadata key")
    parser.add_argument("--compact", action="store_true", help="compact json output")
    args = parser.parse_args()

    rms = load_rms_from_onnx(args.onnx_path, args.key)
    if args.compact:
        print(json.dumps(rms, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(rms, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()