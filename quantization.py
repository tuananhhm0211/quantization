from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from huggingface_hub import login, HfApi
from transformers import file_utils
import os

access_token="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"
login(token = access_token, add_to_git_credential=True)

model_id = "VietAI/vit5-large-vietnews-summarization"
model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if os.path.exists(file_utils.default_cache_path):
  os.rmdir(file_utils.default_cache_path)
else:
  print("The file does not exist")


save_dir = "./model"
optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = OptimizationConfig(
    optimization_level=2,
    enable_transformers_specific_optimizations=True,
    optimize_for_gpu=False,
)

optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)

api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path="./model",
    repo_id="tuananhhmx4/vi5Onnxruntime",
    repo_type="model",
)

