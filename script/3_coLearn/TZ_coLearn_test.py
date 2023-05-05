from function import * 

batch_size= 16
# learning_rate = 1.5e-6
# n_epoch = 40

train_data = TCGADataset_mo("/gpfs/home/chenz05/DL2023/input/metadata/mo_train_dataset.csv",train_transform)
valid_data = TCGADataset_mo("/gpfs/home/chenz05/DL2023/input/metadata/mo_valid_dataset.csv",validation_transform)
test_data = TCGADataset_mo("/gpfs/home/chenz05/DL2023/input/metadata/mo_test_dataset.csv",validation_transform)
import torchvision.models as models

##n_class
# n_class = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/train_dataset.csv")['label']
# n_class = len(numpy.unique(n_class))

###weighted cross entroypy
# weight = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/train_dataset.csv")['label']
# weight =  LabelEncoder().fit_transform(weight)
# weight = numpy.unique(weight,return_counts = True)[1]
# weight = torch.tensor(1 / weight).cuda()
# weight = weight.float()
# criterion = torch.nn.CrossEntropyLoss(weight= weight)
# criterion_1 = torch.nn.BCELoss()

#vgg_model = models.vgg19(pretrained=True)
#vgg_model.classifier[-1]=torch.nn.Linear(in_features = 4096,out_features=n_class,bias=True)
# model = torch.nn.Sequential(vgg_model,
#                             torch.nn.Softmax(dim=0))
#model = vgg_model
#model = model.cuda()

##  Training
train_dataloader = DataLoader(train_data ,batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data ,batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
#model_trained,train_correct_set,valid_correct_set,train_loss_set,valid_loss_set = performance_fn(model, batch_size, learning_rate,criterion = criterion, n_epoch  = n_epoch, train_data = train_data, valid_data  = valid_data  )

model_trained = torch.load("/gpfs/home/chenz05/DL2023/output/3_coLearn/0.3_3_co_resnetwhole_model_18_1e-06.pt")
model_trained.eval()

##  Train
#model_trained.eval()
y_pred, y_true = predict_co_fn(model_trained, train_dataloader)
y_pred_label = copy.deepcopy(y_pred).squeeze()
y_pred_label[y_pred_label > 0.5] = 1
y_pred_label[y_pred_label < 0.5] = 0 
cm = confusion_matrix(y_pred_label,y_true)
print('Training confusion matrix:')
print(cm)
train_AUC, train_fpr, train_tpr = co_marco_AUC_performace(y_pred,y_true)
print('Training AUC = {trainAUC}'.format(trainAUC = train_AUC))
print(train_fpr)
plot_ROC(train_fpr, train_tpr)
plt.savefig("/gpfs/home/chenz05/DL2023/output/3_coLearn/Train_ROC.png")

##  Validation
#model_trained.eval()
y_pred, y_true = predict_co_fn(model_trained, valid_dataloader )
y_pred_label = copy.deepcopy(y_pred).squeeze()
y_pred_label[y_pred_label > 0.5] = 1
y_pred_label[y_pred_label < 0.5] = 0
cm = confusion_matrix(y_pred_label,y_true)
#cm = confusion_matrix(y_pred.argmax(1),y_true)
print('Validation confusion matrix:')
print(cm)
valid_AUC, valid_fpr, valid_tpr = co_marco_AUC_performace(y_pred,y_true)
print('Validation AUC = {validAUC}'.format(validAUC = valid_AUC))
plot_ROC(valid_fpr, valid_tpr)
plt.savefig("/gpfs/home/chenz05/DL2023/output/3_coLearn/Valid_ROC.png")

## Test
#model_trained.eval()
y_pred, y_true = predict_co_fn(model_trained, test_dataloader )
y_pred_label = copy.deepcopy(y_pred).squeeze()
y_pred_label[y_pred_label > 0.5] = 1
y_pred_label[y_pred_label < 0.5] = 0
cm = confusion_matrix(y_pred_label,y_true)
#cm = confusion_matrix(y_pred.argmax(1),y_true)
print('Test confusion matrix:')
print(cm)
test_AUC, test_fpr, test_tpr = co_marco_AUC_performace(y_pred,y_true)
print('Test AUC = {testAUC}'.format(testAUC = test_AUC))
plot_ROC(test_fpr, test_tpr)
plt.savefig("/gpfs/home/chenz05/DL2023/output/3_coLearn/Test_ROC.png")

#epochs  = numpy.arange(0,n_epoch ) + 1 
#plt.figure(figsize=(8,8), dpi=80)
#plt.figure(1)
#ax1 = plt.subplot(221)
#ax1.plot(epochs ,train_correct_set)
#ax2 = plt.subplot(222)
#ax2.plot(epochs ,valid_correct_set)
#ax3 = plt.subplot(223)
#ax3.plot(epochs ,train_loss_set)
#ax4 = plt.subplot(224)
#ax4.plot(epochs ,valid_loss_set)
#plt.savefig("/gpfs/home/chenz05/DL2023/output/1_image/TZ/TZ_VGGwhole_performance_{b}_{l}.png".format(b = batch_size, l = learning_rate))
#plt.clf()

#torch.save(model_trained, "/gpfs/home/chenz05/DL2023/output/1_image/TZ/TZ_1_image_VGGwhole_model.pt")



