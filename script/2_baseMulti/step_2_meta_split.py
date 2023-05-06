import sys
sys.path.append("script/2_baseMulti")
from function import *

CNV_df = nonimage_path_generater_fn("/gpfs/home/chenz05/DL2023/input/CopyNumVar","gene_level_copy_number.v36.tsv",16,"CNV")
RNA_df = nonimage_path_generater_fn("/gpfs/home/chenz05/DL2023/input/RNA","augmented_star_gene_counts.tsv",16,"RNA")
meth_df = nonimage_path_generater_fn("/gpfs/home/chenz05/DL2023/input/methylation","methylation_array.sesame.level3betas.txt",16,"meth")
image_dir = "/gpfs/home/chenz05/DL2023/input/image_norm/"

#####step_2 image and meta
metadata = pd.read_csv('./input/metadata/clinical.csv')
sample = pathlib.Path(image_dir)
sample = list(sample.iterdir())
image_path = [ list(x.iterdir()) for x in sample] 
sample_name = [ re.sub(image_dir, "",str(x)) for x in sample]
sample_name = [ re.sub(r"(-01Z)|(-02Z)", "",str(x)) for x in sample_name ]
df = pd.DataFrame(image_path,index = sample_name )
label =[ metadata["years_to_dead"][metadata['case_submitter_id'] == x].iloc[0] for x in sample_name ]

  ###convert to alive vs death
label = [ "alive" if x == "alive"  else "dead" for x in label]
df['label'] = label
df['sample'] = sample_name
df = pd.melt(df, id_vars=['sample',"label"],var_name='image_index' ,value_name='image_path')
df.index = df["sample"] + "_" + df["image_index"].astype(str)
df = df.drop("image_index",axis = 1)
image_df = df


####step_3 concatenate data frames
list_df = [image_df, CNV_df, RNA_df,meth_df ]
df = pd.concat(list_df, axis=1) 
df = df.loc[~df.isna().apply(any,axis=1)]

####step_4 split train,test, and valid##
id_list = numpy.unique(df["sample"]).tolist()
random.shuffle(id_list)
train_range = int(numpy.ceil(len(id_list) * 0.7))
valid_range = int(numpy.ceil(len(id_list) * 0.9))
train_index = [ numpy.where(df["sample"] == x)[0]  for x in id_list[:train_range] ] 
train_index = [ xx for x in train_index for xx in x ]
valid_index = [ numpy.where(df["sample"] == x)[0]  for x in id_list[train_range:valid_range] ]
valid_index = [ xx for x in valid_index for xx in x ]
test_index = [ numpy.where(df["sample"] == x)[0]   for x in id_list[valid_range:] ]
test_index = [ xx for x in test_index for xx in x ]

train_df = df.iloc[train_index] 
test_df = df.iloc[test_index]
valid_df = df.iloc[valid_index]

train_df.to_csv("./input/metadata/mo_train_dataset.csv")
test_df.to_csv("./input/metadata/mo_test_dataset.csv")
valid_df.to_csv("./input/metadata/mo_valid_dataset.csv")


