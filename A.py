import pandas as pd

df = pd.read_json("hf://datasets/garyzsu/custom_gym_dataset/train.jsonl", lines=True)

print(df.head())

