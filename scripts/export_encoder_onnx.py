# One-time export: MobileSAM TinyViT image_encoder -> ONNX for onnxruntime-web.
#
# Canonical artifact (must match CLIENT_ENCODING.MODEL_PATH):
#   interactive_gallery/public/models/mobile_sam_encoder.onnx
#
# Prerequisite (dev machine only, not a service runtime dep):
#   pip install onnx
#
# HARD REQUIREMENTS (learned 2026-10-05, do not "simplify"):
# - dynamo=False (legacy TorchScript exporter). torch>=2.7's dynamo exporter
#   IGNORES opset_version<18 ("lower version than we have implementations
#   for"), emits opset 18, and its Pad-18 nodes abort onnxruntime-web 1.17's
#   WASM session creation with a bare numeric code. Legacy honors opset 16.
# - NO onnxconverter_common fp16 pass: it upgrades the graph back to opset 18
#   (same Pad failure). Ship fp32 (~28MB, one-time cached download).
# - After export, verify ALL THREE before committing the artifact:
#     1. opset is really 16 (dynamo silently upgrades; legacy honors it).
#     2. single file, zero external_data refs (browser fetches one URL).
#     3. parity: onnxruntime CPU cosine vs torch > 0.999.
import os
import sys
import torch
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "vendor", "MobileSAM"))

from mobile_sam import sam_model_registry

CANONICAL_NAME = "mobile_sam_encoder.onnx"


def export_onnx():
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "models", "mobile_sam.pt")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "interactive_gallery", "public", "models")
    output_path = os.path.join(out_dir, CANONICAL_NAME)

    os.makedirs(out_dir, exist_ok=True)

    sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    encoder = sam.image_encoder
    encoder.eval()

    dummy = torch.randn(1, 3, 1024, 1024, dtype=torch.float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(
            encoder,
            dummy,
            output_path,
            export_params=True,
            verbose=False,
            opset_version=16,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["image_embeddings"],
            dynamo=False,
        )

    import onnx

    model = onnx.load(output_path, load_external_data=False)
    opsets = [(o.domain or "ai.onnx", o.version) for o in model.opset_import]
    assert all(v == 16 for _, v in opsets), f"opset drifted, refusing artifact: {opsets}"
    refs = sum(1 for t in model.graph.initializer if len(t.external_data) > 0)
    assert refs == 0, f"{refs} external_data refs, refusing artifact"
    onnx.checker.check_model(model)
    print(f"Verified canonical artifact: {output_path} (opset 16, single file)")

    # The legacy exporter may stage an orphan sibling .data file; remove it so
    # the browser fetches a single self-contained artifact.
    orphan = output_path + ".data"
    if os.path.exists(orphan):
        os.remove(orphan)
        print(f"Removed orphan staging file: {orphan}")

    print("Done!")


if __name__ == "__main__":
    export_onnx()
