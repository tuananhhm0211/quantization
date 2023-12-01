from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig

model_id = "VietAI/vit5-large-vietnews-summarization"
model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)



save_dir = "./"
optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = OptimizationConfig(
    optimization_level=2,
    enable_transformers_specific_optimizations=True,
    optimize_for_gpu=False,
)

optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
optimized_model = ORTModelForSeq2SeqLM.from_pretrained(save_dir)
