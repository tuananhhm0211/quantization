from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from optimum.onnxruntime import ORTOptimizer, OptimizationConfig
from huggingface_hub import login, HfApi
from transformers import file_utils
import os
import shutil

access_token="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"
login(token = access_token, add_to_git_credential=True)

model_id = "VietAI/vit5-large-vietnews-summarization"
model = ORTModelForSeq2SeqLM.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

save_dir = "./model"
model.save_pretrained(save_dir)

api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path="./model",
    repo_id="tuananhhmx4/vi5Onnxruntime",
    repo_type="model",
)

