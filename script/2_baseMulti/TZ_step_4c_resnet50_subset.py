from function import * 

batch_size = 16
learning_rate = 1e-6
n_epoch = 30

# train_data = pd.read_csv("./input/metadata/mo_train_dataset.csv")
# valid_data = pd.read_csv("./input/metadata/mo_valid_dataset.csv")
# np.random.seed(20230328)
# index_train = np.random.choice(range(len(train_data)), 300, replace=False)
# index_valid = np.random.choice(range(len(valid_data)), 100, replace=False)
# train_data.iloc[index_train].to_csv("./input/metadata/mo_train_dataset_subset.csv")
# valid_data.iloc[index_valid].to_csv("./input/metadata/mo_valid_dataset_subset.csv")

train_data = TCGADataset_mo("/gpfs/home/chenz05/DL2023/input/metadata/mo_train_dataset.csv",train_transform)
valid_data = TCGADataset_mo("/gpfs/home/chenz05/DL2023/input/metadata/mo_valid_dataset.csv",validation_transform)
import torchvision.models as models

##n_class
n_class = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/mo_train_dataset.csv")['label']
n_class = len(numpy.unique(n_class))

##weighted cross entroypy
weight = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/train_dataset.csv")['label']
weight =  LabelEncoder().fit_transform(weight)
weight = numpy.unique(weight,return_counts = True)[1]
weight = torch.tensor(1 / weight).cuda()
weight = weight.float()
criterion = torch.nn.CrossEntropyLoss(weight= weight)

model = resnet50_mo_print(5000)
model = model.cuda()
train_dataloader = DataLoader(train_data ,batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data ,batch_size=batch_size, shuffle=True)
model_trained,train_correct_set,valid_correct_set,train_loss_set,valid_loss_set = performance_fn(model, batch_size,learning_rate,criterion = criterion, n_epoch  = n_epoch, train_data = train_data, valid_data  = valid_data  )
y_pred, y_true = predict_fn(model_trained, train_dataloader ) 
cm = confusion_matrix(y_pred.argmax(1),y_true)
print(cm)
train_AUC = marco_AUC_performace(y_pred,y_true)[0]
y_pred, y_true = predict_fn(model_trained, valid_dataloader )
cm = confusion_matrix(y_pred.argmax(1),y_true)
print(cm)
valid_AUC = marco_AUC_performace(y_pred,y_true)[0]

epochs  = numpy.arange(0,n_epoch ) + 1 
plt.figure(figsize=(8,8), dpi=80)
plt.figure(1)
ax1 = plt.subplot(221)
ax1.plot(epochs ,train_correct_set)
ax2 = plt.subplot(222)
ax2.plot(epochs ,valid_correct_set)
ax3 = plt.subplot(223)
ax3.plot(epochs ,train_loss_set)
ax4 = plt.subplot(224)
ax4.plot(epochs ,valid_loss_set)
plt.savefig("/gpfs/home/chenz05/DL2023/output/2_baseMulti/TZ_2_mo_multiomics_performance_{b}_{l}.png".format(b = batch_size, l = learning_rate ))
plt.clf()




