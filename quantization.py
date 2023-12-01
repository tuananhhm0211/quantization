from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from huggingface_hub import login

access_token="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"
login(token = access_token)

model_id = "VietAI/vit5-large-vietnews-summarization"
model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

access_token="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"
model.push_to_hub("vi5Onnxruntime")
