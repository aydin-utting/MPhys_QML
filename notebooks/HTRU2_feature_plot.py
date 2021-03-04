import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
df = pd.read_csv('../data/HTRU_2.csv')


cols = ["Mean of the integrated profile",
"Standard deviation of the integrated profile",
"Excess kurtosis of the integrated profile",
"Skewness of the integrated profile",
"Mean of the DM-SNR curve",
"Standard deviation of the DM-SNR curve",
"Excess kurtosis of the DM-SNR curve",
"Skewness of the DM-SNR curve",
"Species"]

df.columns = cols


plot = sns.pairplot(df, hue="Species")
plot.savefig('../figures/HTRU2_feature_matrix')