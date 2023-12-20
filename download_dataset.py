import pandas as pd
import os
from bs4 import BeautifulSoup
import urllib.request
import tarfile
from io import BytesIO


def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


dataset_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# Download the dataset
print("Downloading dataset...")
with urllib.request.urlopen(dataset_url) as response:
    tar_data = BytesIO(response.read())

with tarfile.open(fileobj=tar_data, mode="r:gz") as tar:
    tar.extractall(path=".")


# Parse data
print("Parsing data...")
TRAIN_DIR = 'aclImdb/train/'
train = []
for f in os.listdir(os.path.join(TRAIN_DIR, 'pos')):
    with open(os.path.join(TRAIN_DIR, 'pos', f), 'r') as file:
        text = remove_html_tags(file.read())
        train.append({
            "text": text,
            "text_len": len(text),
            "score": int(f.split('.')[0].split('_')[1]),
            "label": 1
            })

for f in os.listdir(os.path.join(TRAIN_DIR, 'neg')):
    with open(os.path.join(TRAIN_DIR, 'neg', f), 'r') as file:
        text = remove_html_tags(file.read())
        train.append({
            "text": text,
            "text_len": len(text),
            "score": int(f.split('.')[0].split('_')[1]),
            "label": 0
            })

df_train = pd.DataFrame(train)
df_train.head()


TEST_DIR = 'aclImdb/test/'
test = []
for f in os.listdir(os.path.join(TEST_DIR, 'pos')):
    with open(os.path.join(TEST_DIR, 'pos', f), 'r') as file:
        test.append({
            "text": remove_html_tags(file.read()),
            "score": int(f.split('.')[0].split('_')[1]),
            "label": 1
            })

for f in os.listdir(os.path.join(TEST_DIR, 'neg')):
    with open(os.path.join(TEST_DIR, 'neg', f), 'r') as file:
        test.append({
            "text": remove_html_tags(file.read()),
            "score": int(f.split('.')[0].split('_')[1]),
            "label": 0
            })

df_test = pd.DataFrame(test)
df_test.head()

# Save to csv
print("Saving to csv...")
df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)

print("Done!")