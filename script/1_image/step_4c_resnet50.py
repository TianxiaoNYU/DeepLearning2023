
import sys
sys.path.append("./script/1_image/")
from function import * 

##================================================================
##                  subset data to find the parameter            =
##================================================================
n_epoch = 100
train_data = pd.read_csv("./input/metadata/train_dataset.csv")
valid_data = pd.read_csv("./input/metadata/valid_dataset.csv")
train_data = TCGADataset_image("./input/metadata/train_dataset.csv",train_transform)
valid_data = TCGADataset_image("./input/metadata/valid_dataset.csv",validation_transform)
import torchvision.models as models

##n_class
n_class = pd.read_csv("./input/metadata/train_dataset.csv")['label']
n_class = len(numpy.unique(n_class))

###weighted cross entroypy
weight = pd.read_csv("./input/metadata/train_dataset.csv")['label']
weight =  LabelEncoder().fit_transform(weight)
weight = numpy.unique(weight,return_counts = True)[1]
weight = torch.tensor(1 / weight).cuda()
weight = weight.float()
criterion = torch.nn.CrossEntropyLoss(weight= weight)


model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(in_features=2048, out_features=n_class, bias=True)
model = model.cuda()
batch_size = 16
learning_rate = 1e-5
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
print(train_AUC)
print(valid_AUC)
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
plt.savefig("./output/1_image/1_image_resnet_wholedata_performance_{b}_{l}.png".format(b = batch_size, l = learning_rate ))
plt.clf()

torch.save(model_trained, "./output/1_image/1_image_resnetwhole_model.pt")


