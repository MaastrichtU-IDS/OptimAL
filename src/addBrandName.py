

import pandas as pd

label_df = pd.read_csv('../data/XMLProduct_ADR.csv')

annot_df = pd.read_csv('../data/XMLProduct_ADR_DBID_Annotations.csv')


label_df = label_df[['Label_ID','Drug_Brand_Name']]

annot_df = annot_df.merge(label_df, on=['Label_ID'])
annot_df.to_csv('../data/XMLProduct_ADR_DBID_Annotations_new.csv')