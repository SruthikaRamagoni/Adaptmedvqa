from inference import run_inference

image_path = "MRI-Brain-Scan-Images.jpg"
question = "Is there any tumor?"

# 1. Zero-shot Qwen2-VL-2B
result1 = run_inference(
    image_path=image_path,
    question=question,
    checkpoint_path=None,           # zero-shot
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    device="cuda"
)
print("Zero-shot Qwen2-VL-2B:", result1)

# 2. Your trained model (manual LoRA baseline)
result2 = run_inference(
    image_path=image_path,
    question=question,
    checkpoint_path="artifacts/checkpoints/Qwen2-VL-2B",  # your checkpoint
    device="cuda"
)
print("Manual LoRA (your model):", result2)
