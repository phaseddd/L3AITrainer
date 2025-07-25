import torch
import pandas as pd
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型结构和权重（ALBERT）
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=2)
model.load_state_dict(torch.load("C:/Project/2/2-2model_test.bin", map_location=device))
model.to(device)
model.eval()

# tokenizer初始化（ALBERT）
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 测试数据处理
test_data = pd.read_csv("C:/Project/2/清洗后邮件数据_test.csv")
test_texts = test_data['text'].tolist()
test_labels = test_data['label'].tolist()

test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len=128)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# 预测
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 计算指标
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# 保存结果
with open('C:/Project/2/model_test_result.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")

# 错误样本分析
error_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true != pred]
error_samples = test_data.iloc[error_indices]
error_samples.to_csv("C:/Project/2/error_analysis.txt", index=False)

print("Finished Testing.")
