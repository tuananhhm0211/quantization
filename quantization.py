from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig

model_id = "VietAI/vit5-large-vietnews-summarization"
model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

model.push_to_hub("tuananhhmx4/vOnnxruntime")
