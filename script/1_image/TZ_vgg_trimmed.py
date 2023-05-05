from function import * 

batch_size= 16
learning_rate = 1e-6
n_epoch = 50

train_data = TCGADataset_image_TZ("/gpfs/home/chenz05/DL2023/input/metadata/train_dataset_trimmed.csv",train_transform)
valid_data = TCGADataset_image_TZ("/gpfs/home/chenz05/DL2023/input/metadata/valid_dataset_trimmed.csv",validation_transform)
import torchvision.models as models

##n_class
n_class = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/train_dataset_trimmed.csv")['years_to_dead']
n_class = len(numpy.unique(n_class))

###weighted cross entroypy
weight = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/train_dataset_trimmed.csv")['years_to_dead']
weight =  LabelEncoder().fit_transform(weight)
weight = numpy.unique(weight,return_counts = True)[1]
weight = torch.tensor(1 / weight).cuda()
weight = weight.float()
criterion = torch.nn.CrossEntropyLoss(weight= weight)
# criterion_1 = torch.nn.BCELoss()

vgg_model = models.vgg19(pretrained=True)
vgg_model.classifier[-1]=torch.nn.Linear(in_features = 4096,out_features=n_class,bias=True)
# model = torch.nn.Sequential(vgg_model,
#                             torch.nn.Softmax(dim=0))
model = vgg_model
model = model.cuda()

train_dataloader = DataLoader(train_data ,batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data ,batch_size=batch_size, shuffle=True)
model_trained,train_correct_set,valid_correct_set,train_loss_set,valid_loss_set = performance_fn(model, batch_size, learning_rate,criterion = criterion, n_epoch  = n_epoch, train_data = train_data, valid_data  = valid_data  )

##  Training
y_pred, y_true = predict_fn(model_trained, train_dataloader) 
cm = confusion_matrix(y_pred.argmax(1),y_true)
print('Training confusion matrix:')
print(cm)
train_AUC = marco_AUC_performace(y_pred,y_true)[0]
print('Training AUC = {trainAUC}'.format(trainAUC = train_AUC))

##  Validation
y_pred, y_true = predict_fn(model_trained, valid_dataloader )
cm = confusion_matrix(y_pred.argmax(1),y_true)
print('Validation confusion matrix:')
print(cm)
valid_AUC = marco_AUC_performace(y_pred,y_true)[0]
print('Validation AUC = {validAUC}'.format(validAUC = valid_AUC))

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
plt.savefig("./output/1_image/TZ".format(b = batch_size, l = learning_rate))
plt.clf()





