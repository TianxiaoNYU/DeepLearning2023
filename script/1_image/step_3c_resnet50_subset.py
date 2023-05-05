import sys
sys.path.append("./script/1_image/")
from function import * 

##================================================================
##                  subset data to find the parameter            =
##================================================================
batch_size_set = [5,16,32]
learning_rate_set = [0.000001,0.00001,0.0001,0.001,0.005]
n_epoch = 100
train_data = pd.read_csv("./input/metadata/train_dataset.csv")
valid_data = pd.read_csv("./input/metadata/valid_dataset.csv")
np.random.seed(20230328)
index_train = np.random.choice(range(len(train_data)), 300, replace=False)
index_valid = np.random.choice(range(len(valid_data)), 100, replace=False)
train_data.iloc[index_train].to_csv("./input/metadata/train_dataset_subset.csv")
valid_data.iloc[index_valid].to_csv("./input/metadata/valid_dataset_subset.csv")

train_data = TCGADataset_image("./input/metadata/train_dataset_subset.csv",train_transform)
valid_data = TCGADataset_image("./input/metadata/valid_dataset_subset.csv",validation_transform)
import torchvision.models as models
df_train = np.zeros((len(batch_size_set),len(learning_rate_set)))
df_valid = np.zeros((len(batch_size_set),len(learning_rate_set)))

##n_class
n_class = pd.read_csv("./input/metadata/train_dataset_subset.csv")['label']
n_class = len(numpy.unique(n_class))

###weighted cross entroypy
weight = pd.read_csv("./input/metadata/train_dataset_subset.csv")['label']
weight =  LabelEncoder().fit_transform(weight)
weight = numpy.unique(weight,return_counts = True)[1]
weight = torch.tensor(1 / weight).cuda()
weight = weight.float()
criterion = torch.nn.CrossEntropyLoss(weight= weight)

for b in range(len(batch_size_set)):
  for l in range(len(learning_rate_set)):
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(in_features=2048, out_features=n_class, bias=True)
    model = model.cuda()
    batch_size = batch_size_set[b]
    learning_rate = learning_rate_set[l]
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
    df_train[b,l] = train_AUC
    df_valid[b,l] = valid_AUC
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
    plt.savefig("./output/1_image/1_image_resnet_performance_{b}_{l}.png".format(b = batch_size, l = learning_rate ))
    plt.clf()

df_train = pd.DataFrame( df_train, index = batch_size_set,columns = learning_rate_set)
df_train.to_csv("./output/1_image/1_image_resnet_train_AUC.csv")
df_valid = pd.DataFrame( df_valid, index = batch_size_set,columns = learning_rate_set)
df_valid.to_csv("./output/1_image/1_image_resnet_valid_AUC.csv")



