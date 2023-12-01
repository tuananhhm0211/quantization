from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from transformers import PushToHubCallback

save_dir = "./model"
model_id = "VietAI/vit5-large-vietnews-summarization"
ORTModelForSeq2SeqLM.from_pretrained(save_dir=save_dir, model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

push_to_hub_callback = PushToHubCallback(
    output_dir=save_dir, tokenizer=tokenizer, hub_model_id="tuananhhmx4/vi5Onnxruntime"
)
