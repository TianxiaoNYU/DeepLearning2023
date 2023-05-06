from function import * 

batch_size = 16
learning_rate = 1e-6
n_epoch = 20


train_data = TCGADataset_mo("/gpfs/home/chenz05/DL2023/input/metadata/mo_train_dataset.csv",train_transform)
valid_data = TCGADataset_mo("/gpfs/home/chenz05/DL2023/input/metadata/mo_valid_dataset.csv",validation_transform)
test_data = TCGADataset_mo("/gpfs/home/chenz05/DL2023/input/metadata/mo_test_dataset.csv",validation_transform)
import torchvision.models as models

##n_class
n_class = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/mo_train_dataset.csv")['label']
n_class = len(numpy.unique(n_class))

##weighted cross entroypy
weight = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/mo_train_dataset.csv")['label']
weight =  LabelEncoder().fit_transform(weight)
weight = numpy.unique(weight,return_counts = True)[1]
weight = torch.tensor(1 / weight).cuda()
weight = weight.float()
criterion = torch.nn.CrossEntropyLoss(weight= weight)

# model = resnet50_mo_print(7048)
# model1 = models.resnet50(pretrained=True)
# model1.fc = torch.nn.Linear(in_features=2048, out_features=n_class, bias=True)
# model1 = model1.cuda()

# model2 = multiomics_FC(dim_mooutput = n_class)
# model2 = model2.cuda()

def CoopLoss(pred1, pred2, target, rho = 0, weight = None):
	tmp_loss = torch.nn.functional.cross_entropy(pred1+pred2, target, weight = weight)
	loss = torch.mean(0.5 * tmp_loss**2 + rho*0.5*(pred1 - pred2)**2)
    # loss = torch.mean((output - target)**2)
	return loss


train_dataloader = DataLoader(train_data ,batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_data ,batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data ,batch_size=batch_size, shuffle=True)

# rho_list = [x * 0.1 for x in range(0, 11)]
rho_list = [0.2, 0.7]

for rho_given in rho_list:

	model1 = models.resnet50(pretrained=True)
	model1.fc = torch.nn.Linear(in_features=2048, out_features=n_class, bias=True)
	model1 = model1.cuda()
	model2 = multiomics_FC(dim_mooutput = n_class)
	model2 = model2.cuda()

	model1_trained, model2_trained, train_correct_set,valid_correct_set,train_loss_set,valid_loss_set = performance_fn2(
	model1, 
	model2,
	batch_size,
	learning_rate,
	criterion = CoopLoss, 
	rho = rho_given,
	weight= weight,
	n_epoch  = n_epoch, 
	train_data = train_data, 
	valid_data  = valid_data)

	torch.save(model1_trained, "/gpfs/home/chenz05/DL2023/output/3_coLearn/TZ/TZ_3_coLearn_resnetwhole_model1_{r}.pt".format(r = rho_given))
	torch.save(model2_trained, "/gpfs/home/chenz05/DL2023/output/3_coLearn/TZ/TZ_3_coLearn_resnetwhole_model2_{r}.pt".format(r = rho_given))

	y_pred, y_true = predict_fn2(model1_trained, model2_trained, train_dataloader ) 
	cm = confusion_matrix(y_pred.argmax(1),y_true)
	print(cm)
	train_AUC = marco_AUC_performace(y_pred,y_true)[0]
	print("Train AUC = {a}".format(a = train_AUC))
	y_pred, y_true = predict_fn2(model1_trained, model2_trained, valid_dataloader )
	cm = confusion_matrix(y_pred.argmax(1),y_true)
	print(cm)
	valid_AUC = marco_AUC_performace(y_pred,y_true)[0]
	print("Validation AUC = {a}".format(a = valid_AUC))

	y_pred, y_true = predict_fn2(model1_trained, model2_trained, test_dataloader )
	cm = confusion_matrix(y_pred.argmax(1),y_true)
	print(cm)
	test_AUC = marco_AUC_performace(y_pred,y_true)[0]
	print("Test AUC = {a}".format(a = test_AUC))
	from sklearn import metrics
	fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred[:,0], pos_label=1)
	test_AUC_new = metrics.auc(fpr, tpr)
	print("sklearn test AUC = {a}".format(a = test_AUC_new))

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
	plt.savefig("/gpfs/home/chenz05/DL2023/output/3_coLearn/TZ/all_TZ_performance_{b}_{l}_{r}.png".format(b = batch_size, l = learning_rate, r = rho_given))
	plt.clf()
	print("Rho_{r} finished. Go to next...".format(r = rho_given))



