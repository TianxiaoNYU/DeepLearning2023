


import sys
sys.path.append("/gpfs/home/chenz05/DL2023/script/3_coLearn/")
from function import *
#rho_list = [0.1,0.3,0.5,0.7,0.9]
#rho_list = [1,5,10,20,100]
#rho_list = [0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]
#rho_list = [0.001,0.1,0.9]

rho_list = [0.9]
#lamda1_list = [1e-9,1e-8,1e-7,1e-6,1e-5,1e-4]
#lr_list = [1e-4,1e-5,1e-6]
lr_list = [1e-5]
n_epoch = 25

train_data = TCGADataset_mo("/gpfs/home/chenz05/DL2023/input/metadata/mo_train_dataset.csv",train_transform)
valid_data = TCGADataset_mo("/gpfs/home/chenz05/DL2023/input/metadata/mo_valid_dataset.csv",validation_transform)


import torchvision.models as models
df_train = np.zeros((len(rho_list),len(lr_list)))
df_valid = np.zeros((len(rho_list),len(lr_list)))


##n_class
n_class = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/mo_train_dataset.csv")['label']
n_class = len(numpy.unique(n_class))

##weighted cross entroypy
weight = pd.read_csv("./input/metadata/mo_train_dataset.csv")['label']
weight =  LabelEncoder().fit_transform(weight)
weight = numpy.unique(weight,return_counts = True)[1]
weight = torch.tensor(1 / weight).cuda()
weight = weight.float()



for r in range(len(rho_list)):
  for l in range(len(lr_list)):
    print("rho: ",rho_list[r] )
    print("lr: ",lr_list[l])
    rho = rho_list[r]
    lr = lr_list[l]
    criterion = CO_loss(weight = weight,rho = rho,lamda1 =0, lamda2 = 0)
    model = resnet50_co(2048,1000,3000,1000 )
    model = model.cuda()
    batch_size = 12
    learning_rate = lr
    train_dataloader = DataLoader(train_data ,batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data ,batch_size=batch_size, shuffle=True)
    model_trained,train_correct_set,valid_correct_set,train_loss_set,valid_loss_set = co_performance_fn(model, batch_size,learning_rate,criterion = criterion, n_epoch  = n_epoch, train_data = train_data, valid_data  = valid_data  )
    y_pred, y_true = predict_co_fn(model_trained, train_dataloader ) 
    y_pred_label = copy.deepcopy(y_pred).squeeze()
    y_pred_label[y_pred_label  > 0.5] = 1 
    y_pred_label[y_pred_label < 0.5] =0
    cm = confusion_matrix(y_pred_label,y_true)
    print(cm)
    train_AUC = co_marco_AUC_performace(y_pred,y_true)[0]

    y_pred, y_true = predict_co_fn(model_trained, valid_dataloader )
    y_pred_label = copy.deepcopy(y_pred).squeeze()
    y_pred_label[y_pred_label  > 0.5] = 1
    y_pred_label[y_pred_label < 0.5] =0
    cm = confusion_matrix(y_pred_label,y_true)
    print(cm)
    valid_AUC = co_marco_AUC_performace(y_pred,y_true)[0]

    df_train[r,l] = train_AUC
    df_valid[r,l] = valid_AUC
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
    plt.savefig("/gpfs/home/chenz05/DL2023/output/3_coLearn/3_co_resnet_performance_whole_{r}_{l}.png".format(r =rho , l = lr ))
    plt.clf()

df_train = pd.DataFrame( df_train, index = rho_list,columns = lr_list)
df_train.to_csv("/gpfs/home/chenz05/DL2023/output/3_coLearn/3_co_resnet_train_AUC.csv")
df_valid = pd.DataFrame( df_valid, index = rho_list,columns = lr_list)
df_valid.to_csv("/gpfs/home/chenz05/DL2023/output/3_coLearn/3_co_resnet_valid_AUC.csv")






