import pandas as pd
import os

attributes_df = pd.read_csv("./Code/data/img_align_celeba/list_attr_celeba.csv")

celeba_data_path = './Code/data/img_align_celeba/img_align_celeba/img_align_celeba'

print(len(os.listdir(celeba_data_path)))