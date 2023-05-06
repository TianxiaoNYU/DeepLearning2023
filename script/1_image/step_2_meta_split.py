import pandas as pd 
import pathlib
import re
metadata = pd.read_csv('./input/metadata/clinical.csv')
sample = pathlib.Path("/gpfs/home/chenz05/DL2023/input/image_norm")
sample = list(sample.iterdir())
image_path = [ list(x.iterdir()) for x in sample] 
sample_name = [ re.sub("/gpfs/home/chenz05/DL2023/input/image_norm/", "",str(x)) for x in sample]
sample_name = [ re.sub(r"(-01Z)|(-02Z)", "",str(x)) for x in sample_name ]
df = pd.DataFrame(image_path,index = sample_name )
label =[ metadata["years_to_dead"][metadata['case_submitter_id'] == x].iloc[0] for x in sample_name ]

###convert to alive vs death
label = [ "alive" if x == "alive"  else "dead" for x in label]
###
df['label'] = label
df['sample'] = sample_name

######split into train,validation,test### by patients
random_seed = 1234567
train = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.7,random_state = random_seed))
rest = df.loc[~df.index.isin(train.index)]

valid = rest.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=1/3,random_state = random_seed))
test = rest.loc[~rest.index.isin(valid.index)]

###wide to long format
test = pd.melt(test, id_vars=['sample',"label"],var_name='image_index' ,value_name='image_path')
train = pd.melt(train, id_vars=['sample',"label"],var_name='image_index' ,value_name='image_path')
valid = pd.melt(valid, id_vars=['sample',"label"],var_name='image_index' ,value_name='image_path')

train = train.loc[~train["image_path"].isna()]
test = test.loc[~test["image_path"].isna()]
valid = valid.loc[~valid["image_path"].isna()]


test.to_csv("./input/metadata/test_dataset.csv")
valid.to_csv("./input/metadata/valid_dataset.csv")
train.to_csv("./input/metadata/train_dataset.csv")


print( "stats for train:")
print(train["label"].value_counts() )
print( "stats for test:")
print(test["label"].value_counts() )
print( "stats for validation:")
print(valid["label"].value_counts() )


