import os
import sys
import torch
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "vendor", "MobileSAM"))

from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel

def custom_mask_postprocessing(self, masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
    return masks

def export_onnx():
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "models", "mobile_sam.pt")
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "interactive_gallery", "public", "models", "mobile_sam_decoder.onnx")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    onnx_model = SamOnnxModel(sam, return_single_mask=False)
    
    import types
    onnx_model.mask_postprocessing = types.MethodType(custom_mask_postprocessing, onnx_model)
    onnx_model.eval()
    
    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 1, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 1), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1024, 1024], dtype=torch.float),
    }
    
    output_names = ["masks", "iou_predictions", "low_res_masks"]
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        torch.onnx.export(
            onnx_model,
            tuple(dummy_inputs.values()),
            output_path,
            export_params=True,
            verbose=False,
            opset_version=16,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes={
                "point_coords": {1: "num_points"},
                "point_labels": {1: "num_points"},
            },
            fallback=True
        )
        
    print("Done!")

if __name__ == "__main__":
    export_onnx()
