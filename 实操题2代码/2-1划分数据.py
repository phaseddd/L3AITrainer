import pandas as pd
import re

import os
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv("")

# 数据清洗
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除特殊符号
    text = text.lower()  # 转换为小写
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # 去除停用词
    return text

data['text'] = data['text'].apply(clean_text)

# 数据划分
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# 保存结果
os.makedirs("", exist_ok=True)
train_data.to_csv("", index=False)
test_data.to_csv("", index=False)
