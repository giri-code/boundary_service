import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("../interactive_gallery/public/models/mobile_sam_decoder.onnx")
for i in sess.get_inputs():
    print(i.name, i.shape)
for o in sess.get_outputs():
    print(o.name, o.shape)

image_embeddings = np.random.randn(1, 256, 64, 64).astype(np.float32)
point_coords = np.array([[[500.0, 500.0]]], dtype=np.float32)
point_labels = np.array([[1.0]], dtype=np.float32)
mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
has_mask_input = np.array([0.0], dtype=np.float32)

inputs = {
    "image_embeddings": image_embeddings,
    "point_coords": point_coords,
    "point_labels": point_labels,
    "mask_input": mask_input,
    "has_mask_input": has_mask_input
}
res = sess.run(None, inputs)
print("Output shapes:")
for r in res:
    print(r.shape)
