import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("../dataset/kidney_disease.csv")
# print(df.columns)

# mapping yes/no & abnormal/normal & present/absent & good/poor & notckd/ckd --> 1/0
df[['htn', 'dm', 'cad', 'pe', 'ane']] = df[['htn', 'dm', 'cad', 'pe', 'ane']].replace(to_replace={'yes': 1, 'no': 0})
df[['rbc', 'pc']] = df[['rbc', 'pc']].replace(to_replace={'abnormal': 1, 'normal': 0})
df[['pcc', 'ba']] = df[['pcc', 'ba']].replace(to_replace={'present': 1, 'notpresent': 0})
df[['appet']] = df[['appet']].replace(to_replace={'good': 1, 'poor': 0, 'no': np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd': 1.0, 'ckd\t': 1.0, 'notckd': 0.0, 'no': 0.0})
df.rename(columns={'classification': 'class'}, inplace=True)

# unchanged values
df['pe'] = df['pe'].replace(to_replace='good', value=0)  # Not having pedal edema is good
df['appet'] = df['appet'].replace(to_replace='no', value=0)
df['cad'] = df['cad'].replace(to_replace='\tno', value=0)
df['dm'] = df['dm'].replace(to_replace={'\tno': 0, '\tyes': 1, ' yes': 1, '': np.nan})

# dropping some null values
print(f"Total rows :: {len(df)}")
# print(f"Classes :: {df['class'].value_counts()}")
df = df.dropna()
print(f"Total rows (cleaned) :: {len(df)}")
# print(f"Classes :: {df['class'].value_counts()}")

# dropping the id column
df.drop('id', axis=1, inplace=True)

# heat map to visualize correlation
corr_df = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_df, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_df, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlations between different predictors')
plt.show()

# saving the final data set
df.to_csv("../dataset/processed_kidney_disease.csv", index=False)
