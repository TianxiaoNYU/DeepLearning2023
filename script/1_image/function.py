from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io
import torch
from skimage import color
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import numpy
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import trange
import copy 

def H_E_Staining(img, Io=240, alpha=1, beta=0.15):
	# define height and width of image
	h, w, c = img.shape
	# reshape image
	img = img.reshape((-1,3))
	# calculate optical density
	OD = -np.log((img.astype(np.float)+1)/Io)
	# remove transparent pixels
	ODhat = OD[~np.any(OD<beta, axis=1)]
	# compute eigenvectors
	eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
	#eigvecs *= -1
	#project on the plane spanned by the eigenvectors corresponding to the two 
	# largest eigenvalues    
	That = ODhat.dot(eigvecs[:,1:3])
	phi = np.arctan2(That[:,1],That[:,0])
	minPhi = np.percentile(phi, alpha)
	maxPhi = np.percentile(phi, 100-alpha)
	vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
	vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
	# a heuristic to make the vector corresponding to hematoxylin first and the 
	# one corresponding to eosin second
	if vMin[0] > vMax[0]:
		HE = np.array((vMin[:,0], vMax[:,0])).T
	else:
		HE = np.array((vMax[:,0], vMin[:,0])).T
	return HE

def normalizeStaining(img, HERef, Io=240, alpha=1, beta=0.15):
	maxCRef = np.array([1.9705, 1.0308])
	# define height and width of image
	h, w, c = img.shape
	# reshape image
	img = img.reshape((-1,3))
	# calculate optical density
	OD = -np.log((img.astype(np.float)+1)/Io)
	# remove transparent pixels
	ODhat = OD[~np.any(OD<beta, axis=1)]
	# compute eigenvectors
	eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
	#eigvecs *= -1
	#project on the plane spanned by the eigenvectors corresponding to the two 
	# largest eigenvalues    
	That = ODhat.dot(eigvecs[:,1:3])
	phi = np.arctan2(That[:,1],That[:,0])
	minPhi = np.percentile(phi, alpha)
	maxPhi = np.percentile(phi, 100-alpha)
	vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
	vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
	# a heuristic to make the vector corresponding to hematoxylin first and the 
	# one corresponding to eosin second
	if vMin[0] > vMax[0]:
		HE = np.array((vMin[:,0], vMax[:,0])).T
	else:
		HE = np.array((vMax[:,0], vMin[:,0])).T
	# rows correspond to channels (RGB), columns to OD values
	Y = np.reshape(OD, (-1, 3)).T
	# determine concentrations of the individual stains
	C = np.linalg.lstsq(HE,Y, rcond=None)[0]
	# normalize stain concentrations
	maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
	tmp = np.divide(maxC,maxCRef)
	C2 = np.divide(C,tmp[:, np.newaxis])
	# recreatefthe image using reference mixing matrix
	Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
	Inorm[Inorm>255] = 254
	Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
	return Inorm

image_ref = io.imread("/gpfs/home/chenz05/DL2023/input/image/TCGA-75-6206-01Z/6144_7168_8.jpg")
HE_ref = H_E_Staining(image_ref)




train_transform = transforms.Compose([
        transforms.ToPILImage(),
       # transforms.CenterCrop(896),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.RandomErasing(p=0.5, scale=(0.1, 0.25), ratio=(0.3, 3.3), value="random")
       ])

validation_transform = transforms.Compose([
        transforms.ToPILImage(),
       # transforms.CenterCrop(896),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ])


class TCGADataset_image(Dataset):
    """Chest X-ray dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        encoder = LabelEncoder()
        self.data_frame['class'] = encoder.fit_transform(self.data_frame['label'])
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame["image_path"][idx]
        image = io.imread(img_name)
    #    image = normalizeStaining(image,HE_ref)
        image = image.transpose((2, 0, 1))   ###from H x W x C to C x H x W  
        image = torch.tensor( image) ##from numpy array to torch tensor
        image_class = self.data_frame['class'][idx]
         
       
        if self.transform:
            image = self.transform(image)
        
        x =  image
        y = image_class
        sample = {"x" : x, "y" : y}
        return sample


def train_fn(model, dataloader, criterion, optimizer  ):
    model.train()
    accum_train_loss = 0
    correct = 0
    for train_sample in dataloader:
        x = train_sample["x"].cuda()
        y = train_sample["y"].cuda()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion( y_pred, y )
        loss.backward()
        optimizer.step()
        accum_train_loss += loss.item()
        correct += sum(y == y_pred.argmax(1)).double()
    average_loss = accum_train_loss/len(dataloader)
    return correct,average_loss, model 

def train_fn_inception(model, dataloader, criterion, optimizer  ):
    model.train()
    accum_train_loss = 0
    correct = 0
    for train_sample in dataloader:
        x = train_sample["x"].cuda()
        y = train_sample["y"].cuda()
        optimizer.zero_grad()
        y_pred,aux_pred = model(x)
        loss_main = criterion( y_pred, y )
        loss_aux =  criterion( aux_pred, y )
        loss = loss_main + 0.4 * loss_aux
        loss.backward()
        optimizer.step()
        accum_train_loss += loss.item()
        correct += sum(y == y_pred.argmax(1)).double()
    average_loss = accum_train_loss/len(dataloader)
    return correct,average_loss, model


def valid_fn(model, dataloader, criterion,optimizer):
   with torch.no_grad():
     model.eval()
     accum_valid_loss = 0
     correct = 0
     y_pred_set = []
     y_true_set = []
     softmax = torch.nn.Softmax(dim = 1) 
     for valid_sample in dataloader:
          x = valid_sample["x"].cuda()
          y = valid_sample["y"].cuda()
          y_pred = model(x)
          loss = criterion( y_pred, y )
          accum_valid_loss += loss.item()
          correct += sum(y == y_pred.argmax(1)).double()
          y_pred_set.append( softmax(model(x)).cpu())
          y_true_set.append(y.cpu())
     average_loss = accum_valid_loss / len(dataloader)
     y_pred_set = [x.detach().numpy() for x in y_pred_set ]
     y_pred_set = numpy.concatenate( y_pred_set , axis=0 )
     y_true_set = numpy.concatenate( y_true_set , axis=0 )
   return correct, average_loss, y_pred_set, y_true_set 

def performance_fn(model,batch_size,learning_rate,train_data,valid_data,criterion,n_epoch = 50):
    train_correct_set = []
    valid_correct_set = []
    train_loss_set = []
    valid_loss_set = []
    criterion = criterion
    optimizer = optim.Adam(model.parameters(),lr = learning_rate )
    valid_loss_best = np.inf
    valid_correct_best = 0
    model_best = None
    for epoch in trange(1, n_epoch+1):
        train_dataloader = DataLoader(train_data ,batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_data ,batch_size=batch_size, shuffle=True)
        train_correct,train_loss, model_trained = train_fn(model,train_dataloader, criterion, optimizer )
        valid_correct,valid_loss,y_valid_pred,y_valid_true = valid_fn(model_trained,valid_dataloader, criterion, optimizer )
        ###loss & acc 
        train_correct = train_correct.cpu()
        valid_correct = valid_correct.cpu()
        train_correct = train_correct.double() / len(train_data)
        valid_correct = valid_correct.double() / len(valid_data)
        print("train acc:",train_correct )
        print("valid acc:",valid_correct ) 
        print("train loss:",train_loss )
        print("valid loss:",valid_loss )
        train_correct_set.append(train_correct)
        valid_correct_set.append(valid_correct)
        train_loss_set.append(train_loss)
        valid_loss_set.append(valid_loss)
        ###confusion matrix
        cm = confusion_matrix(y_valid_pred.argmax(1),y_valid_true)
        print(cm)
        ###best model
        if valid_loss <= valid_loss_best and valid_correct >= valid_correct_best and train_correct > 0.7:  
            model_best = copy.deepcopy(model_trained)
            valid_loss_best = valid_loss * 1.5
            valid_correct_best = valid_correct
    return model_best,train_correct_set,valid_correct_set,train_loss_set,valid_loss_set


def performance_fn_inception(model,batch_size,learning_rate,train_data,valid_data,criterion,n_epoch = 50):
    train_correct_set = []
    valid_correct_set = []
    train_loss_set = []
    valid_loss_set = []
    criterion = criterion
    optimizer = optim.AdamW(model.parameters(),lr = learning_rate )
    valid_loss_best = np.inf
    valid_correct_best = 0
    model_best = None
    for epoch in trange(1, n_epoch+1):
        train_dataloader = DataLoader(train_data ,batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_data ,batch_size=batch_size, shuffle=True)
        train_correct,train_loss, model_trained = train_fn_inception(model,train_dataloader, criterion, optimizer )
        valid_correct,valid_loss,y_valid_pred,y_valid_true = valid_fn(model_trained,valid_dataloader, criterion, optimizer )
        ###loss & acc 
        train_correct = train_correct.cpu()
        valid_correct = valid_correct.cpu()
        train_correct = train_correct.double() / len(train_data)
        valid_correct = valid_correct.double() / len(valid_data)
        print("train acc:",train_correct )
        print("valid acc:",valid_correct )
        print("train loss:",train_loss )
        print("valid loss:",valid_loss )
        train_correct_set.append(train_correct)
        valid_correct_set.append(valid_correct)
        train_loss_set.append(train_loss)
        valid_loss_set.append(valid_loss)
        ###cm
        cm = confusion_matrix(y_valid_pred.argmax(1),y_valid_true)
        print(cm)
        ##best model
        if valid_loss <= valid_loss_best and valid_correct >= valid_correct_best:
            model_best = copy.deepcopy(model_trained) 
            valid_loss_best = valid_loss
            valid_correct_best = valid_correct
    return model_best,train_correct_set,valid_correct_set,train_loss_set,valid_loss_set

def predict_fn(model, dataloader):
    model.eval()
    y_pred = []
    y_true = []
    softmax = torch.nn.Softmax(dim = 1)
    with torch.no_grad():
      for sample in dataloader:
        x = sample["x"].cuda()
        y = sample["y"].cpu()
        y_pred.append(softmax(model(x)).cpu())
        y_true.append(y)
    y_pred = [x.detach().numpy() for x in y_pred ]
    y_pred = numpy.concatenate( y_pred , axis=0 )
    y_true = numpy.concatenate( y_true, axis=0 )
    return y_pred, y_true 


def marco_AUC_performace(y_pred, y_true ):
    n_classes = len(np.unique(y_true))
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
       y_label = (y_true == i) * 1
       fpr[i], tpr[i], _ = roc_curve(y_label , y_pred[:, i])
       roc_auc[i] = auc(fpr[i], tpr[i])
    fpr_grid = np.linspace(0.0, 1.0, 1000)
   # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # Average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return(roc_auc["macro"],fpr["macro"],tpr["macro"] )



def plot_ROC(fpr,tpr):
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic') 
    plt.plot(fpr["macro"], tpr["macro"])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

 
class TCGADataset_image_TZ(Dataset):
    """Chest X-ray dataset from https://nihcc.app.box.com/v/ChestXray-NIHCC."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        encoder = LabelEncoder()
        self.data_frame['class'] = encoder.fit_transform(self.data_frame['label'])
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame["image_path"][idx]
        image = io.imread(img_name)
        # image = normalizeStaining(image,HE_ref)
        image = image.transpose((2, 0, 1))   ###from H x W x C to C x H x W  
        image = torch.tensor( image) ##from numpy array to torch tensor
        image_class = self.data_frame['class'][idx]
         
       
        if self.transform:
            image = self.transform(image)
        
        x =  image
        y = image_class
        sample = {"x" : x, "y" : y}
        return sample

