print("--- BƯỚC 1: LIBRARY INSTALL ---")

!pip install simpletransformers -q

from google.colab import drive
import os
import json
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args

drive.mount('/content/drive')

DRIVE_PATH = '/content/drive/MyDrive/end term AI proj'
SUMMARIES_DATA_PATH = os.path.join(DRIVE_PATH, 'data', 'processed', 'summaries')
CLEANED_DATA_PATH = os.path.join(DRIVE_PATH, 'data', 'cleaned')
MODEL_SAVE_PATH = os.path.join(DRIVE_PATH, 'models', 'summarizer', 't5-legal-summarizer-final')

print("Cài đặt và thiết lập hoàn tất!")


print("\n--- BƯỚC 2: DATA ---")

data_to_process = []

for filename in os.listdir(SUMMARIES_DATA_PATH):
    if filename.endswith('.json'):
        json_path = os.path.join(SUMMARIES_DATA_PATH, filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

            source_filename = json_data.get("source_file")
            summary_text = json_data.get("summary")

            if source_filename and summary_text:
                txt_path = os.path.join(CLEANED_DATA_PATH, source_filename)
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as txt_f:
                        full_text = txt_f.read()
                        # Thêm cột "prefix"
                        data_to_process.append({
                            "prefix": "summarize",
                            "input_text": full_text,
                            "target_text": summary_text
                        })

# Tạo DataFrame với đúng tên cột yêu cầu
train_df = pd.DataFrame(data_to_process)

print(f"Đã xử lý {len(train_df)} cặp văn bản/tóm tắt.")
print("5 dòng đầu của dữ liệu huấn luyện:")
print(train_df.head())


print("\n--- BƯỚC 3: TRAIN ---")
from simpletransformers.t5 import T5Model, T5Args

# 1. Cấu hình các tham số
model_args = T5Args()
model_args.num_train_epochs = 5
model_args.learning_rate = 2e-5
model_args.overwrite_output_dir = True

model_args.train_batch_size = 2  # Giảm số lượng "pizza" mỗi lần nướng
model_args.eval_batch_size = 2
model_args.max_seq_length = 512 # Giảm kích thước "pizza" (văn bản gốc)
model_args.max_length = 256      # Giữ nguyên kích thước tóm tắt
model_args.gradient_accumulation_steps = 2
# -----------------------------------------------

model_args.output_dir = MODEL_SAVE_PATH

# 2. Tạo mô hình T5Model
model = T5Model(
    model_type="t5",
    model_name="VietAI/vit5-base",
    args=model_args,
    use_cuda=True
)

# 3. Bắt đầu huấn luyện!
model.train_model(train_df)

print("Huấn luyện hoàn tất!")
print(f"✅ Đã lưu mô hình tóm tắt thành công vào: {MODEL_SAVE_PATH}")
