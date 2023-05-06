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
import pathlib
import re
import random
import copy
import torchvision.models as models


def nonimage_path_generater_fn(path, pattern, num_img_copy,omics_type):
    sample = pathlib.Path(path) 
    sample = list(sample.iterdir())
    sample_name = [ re.sub("^.*/","",str(x)) for x in sample ]
    sample_path = [ list(x.iterdir()) for x in sample]
    sample_path = [xx for x in sample_path for xx in x if pattern in str(xx)]
    sample_dict = dict(zip(sample_name,sample_path))
    seed_matrix = numpy.random.randint(0,2**32-1,(len(sample_dict),num_img_copy))
    seed_matrix = pd.DataFrame(seed_matrix)
    seed_matrix.index = sample_name
    seed_matrix[omics_type] = sample_path
    seed_matrix["ID"] = sample_name
    seed_matrix = pd.melt(seed_matrix, id_vars=[omics_type,"ID"],var_name='seed_index' ,value_name= omics_type + "_seed")
    seed_matrix.index = seed_matrix["ID"] + "_" + seed_matrix["seed_index"].astype(str)
    seed_matrix = seed_matrix.drop(["ID","seed_index"],axis = 1 )
    return seed_matrix


class topK_CV_help_fn(Dataset):
    def __init__(self, dataframe):
        self.data_frame = dataframe
   
    def __len__(self):
        return len(self.data_frame)    

    def __getitem__(self, idx):
        cnv_path = self.data_frame["CNV"][idx]
        cnv = pd.read_csv(cnv_path, delimiter = '\t')
        cnv = numpy.array(cnv["copy_number"])
        cnv[np.isnan(cnv)] = 0 

        meth_path = self.data_frame["meth"][idx]
        meth = pd.read_csv(meth_path, delimiter = '\t',header = None)
        meth = np.array(meth.iloc[:,1])
        meth[np.isnan(meth)] = 0        

        RNA_path = self.data_frame["RNA"][idx]
        RNA = pd.read_csv(RNA_path,delimiter = "\t",comment="#")
        RNA = numpy.array(RNA["unstranded"])[4:]
        RNA[np.isnan(RNA)] = 0        

        sample = {"CNV" : cnv, "meth" : meth,"RNA" : RNA}
        return sample

def row_CV_fn(df,base_mean ):
    rm = df.mean(axis=1) 
    rs = df.std(axis=1)
    rs[rs < 0.01] = 0
    return rs/(rm - base_mean)

def OneD_omics_normalization_fn(df):
    df = (df.T - df.mean(axis =1)) / df.std(axis = 1)
    return df.T

def topK_CV_fn(RNA_num, meth_num,CNV_num,normalization = True  ):
    CNV_df = nonimage_path_generater_fn("/gpfs/home/chenz05/DL2023/input/CopyNumVar","gene_level_copy_number.v36.tsv",1,"CNV")
    RNA_df = nonimage_path_generater_fn("/gpfs/home/chenz05/DL2023/input/RNA","augmented_star_gene_counts.tsv",1,"RNA")
    meth_df = nonimage_path_generater_fn("/gpfs/home/chenz05/DL2023/input/methylation","methylation_array.sesame.level3betas.txt",1,"meth")
    df = pd.concat([CNV_df["CNV"],RNA_df["RNA"], meth_df["meth"]],axis = 1 )
    dataset = topK_CV_help_fn(df)
    list_CNV = []
    list_meth = []
    list_RNA = []
    for i in range(len(dataset)):
          list_CNV.append(dataset[i]["CNV"])
          list_meth.append(dataset[i]["meth"])
          list_RNA.append(dataset[i]["RNA"])
    
    CNV_df = pd.DataFrame(list_CNV,index = CNV_df.index).T
    CNV_df.columns = CNV_df.columns.map(lambda x: re.sub(r'_0', '', x))
    meth_df = pd.DataFrame(list_meth,index = meth_df.index).T
    meth_df.columns = meth_df.columns.map(lambda x: re.sub(r'_0', '', x))
    meth_df[np.isnan(meth_df)] = 0
    meth_df = meth_df[np.sum(meth_df == 0,axis = 1) < meth_df.shape[1] * 0.5]
    RNA_df = pd.DataFrame(list_RNA,index = RNA_df.index).T
    RNA_df.columns = RNA_df.columns.map(lambda x: re.sub(r'_0', '', x))
    
    top_RNA = np.argsort(-row_CV_fn(RNA_df, 100))[:RNA_num]
    top_meth = np.argsort(-row_CV_fn(meth_df ,0.01 ))[:meth_num]
    top_CNV = np.argsort(-row_CV_fn(CNV_df,1))[:CNV_num]
    
 
    RNA_df = RNA_df.iloc[top_RNA,:]
    meth_df = meth_df.iloc[top_meth,:]
    CNV_df = CNV_df.iloc[top_CNV,:]
    
    if normalization:
       RNA_df = OneD_omics_normalization_fn(RNA_df)
       CNV_df = OneD_omics_normalization_fn(CNV_df)
       meth_df = OneD_omics_normalization_fn(meth_df)   
    

    RNA_df.to_csv("./input/metadata/RNA_top.csv")
    meth_df.to_csv("./input/metadata/meth_top.csv")
    CNV_df.to_csv("./input/metadata/CNV_top.csv")


def Gaussion_noise_fn(data,seed, mean = 0, std = 0.1):
    rng = np.random.RandomState(seed)
    return data + rng.normal(loc=mean, scale=std, size=data.shape)

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
        transforms.RandomErasing(p=0.5, scale=(0.1, 0.25), ratio=(0.3, 3.3), value="random")
     #   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ])

validation_transform = transforms.Compose([
        transforms.ToPILImage(),
       # transforms.CenterCrop(896),
        transforms.ToTensor(),
       # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
       ])


     

class TCGADataset_mo(Dataset):
    def __init__(self, csv_file, transform=None , noise_fn = Gaussion_noise_fn):
        self.data_frame = pd.read_csv(csv_file)
        self.CNV_df = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/CNV_top.csv")
        self.meth_df = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/meth_top.csv")
        self.RNA_df = pd.read_csv("/gpfs/home/chenz05/DL2023/input/metadata/RNA_top.csv")
        encoder = LabelEncoder()
        self.data_frame['class'] = encoder.fit_transform(self.data_frame['label'])
        self.transform = transform
        self.noise_fn = noise_fn

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        ID = self.data_frame["sample"][idx]
        CNV = self.CNV_df[ID]
        CNV_seed = int(self.data_frame["CNV_seed"][idx])
        meth = self.meth_df[ID]
        meth_seed = int(self.data_frame["meth_seed"][idx])
        RNA = self.RNA_df[ID]
        RNA_seed = int(self.data_frame["RNA_seed"][idx])
        
        image_path = self.data_frame["image_path"][idx]
        image = io.imread(image_path)        
        #image = normalizeStaining(image,HE_ref)
        image = image.transpose((2, 0, 1))   ###from H x W x C to C x H x W  
        image = torch.tensor( image) ##from numpy array to torch tensor
        image_class = self.data_frame['class'][idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.noise_fn:
           CNV = self.noise_fn(CNV,CNV_seed)
           RNA = self.noise_fn(RNA,RNA_seed )
           meth = self.noise_fn(meth,meth_seed)

        x =  image
        y = image_class
        sample = {"image" : image,
                  "CNV" : torch.tensor(CNV).float(),
                  "meth" : torch.tensor(meth).float(),
                  "RNA" : torch.tensor(RNA).float(),
                   "y" : torch.tensor(y)}

        return sample


def train_fn(model, dataloader, criterion, optimizer  ):
    model.train()
    accum_train_loss = 0
    correct = 0
    for train_sample in dataloader:
        y = train_sample["y"].cuda()
        image = train_sample["image"].cuda()
        CNV = train_sample["CNV"].cuda()
        meth = train_sample["meth"].cuda()
        RNA = train_sample["RNA"].cuda()
        optimizer.zero_grad()
        y_pred = model(image,CNV,meth,RNA)
        loss = criterion( y_pred, y )
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
          y = valid_sample["y"].cuda()
          image = valid_sample["image"].cuda()
          CNV = valid_sample["CNV"].cuda()
          meth = valid_sample["meth"].cuda()
          RNA = valid_sample["RNA"].cuda()
          y_pred = model(image,CNV,meth,RNA)
          loss = criterion( y_pred, y )
          accum_valid_loss += loss.item()
          correct += sum(y == y_pred.argmax(1)).double()
          y_pred_set.append( softmax(model( image,CNV,meth,RNA )).cpu())
          y_true_set.append(y.cpu())
     average_loss = accum_valid_loss / len(dataloader)
     y_pred_set = [x.detach().numpy() for x in y_pred_set ]
     y_pred_set = numpy.concatenate( y_pred_set , axis=0 )
     y_true_set = numpy.concatenate( y_true_set , axis=0 )
   return correct, average_loss, y_pred_set, y_true_set 

def predict_fn(model, dataloader):
    model.eval()
    y_pred = []
    y_true = []
    softmax = torch.nn.Softmax(dim = 1)
    with torch.no_grad():
      for sample in dataloader:
        y = sample["y"]
        image = sample["image"].cuda()
        CNV = sample["CNV"].cuda()
        meth = sample["meth"].cuda()
        RNA = sample["RNA"].cuda()
        y_pred.append(softmax(model(image,CNV,meth,RNA )).cpu())
        y_true.append(y)
    y_pred = [x.detach().numpy() for x in y_pred ]
    y_pred = numpy.concatenate( y_pred , axis=0 )
    y_true = numpy.concatenate( y_true, axis=0 )
    return y_pred, y_true 


def performance_fn(model,batch_size,learning_rate,train_data,valid_data,criterion,n_epoch = 50):
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

        if valid_loss <= valid_loss_best and valid_correct >= valid_correct_best and train_correct > 0.9 :  
            model_best = copy.deepcopy(model_trained)
            valid_loss_best = valid_loss * 1.5
            valid_correct_best = valid_correct

    return model_best,train_correct_set,valid_correct_set,train_loss_set,valid_loss_set

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





class resnet50_mo(torch.nn.Module):
      def __init__(self,dim_moinput = 7048):
         super(resnet50_mo, self).__init__()
         resnet50_whole = models.resnet50(pretrained=True)
         self.resnet50_avg = torch.nn.Sequential(*list(resnet50_whole.children())[:-1])
         self.MLP_mo = torch.nn.Sequential(
                       torch.nn.Linear(in_features=dim_moinput, out_features= 1024),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(0.2),
                       torch.nn.Linear(in_features=1024, out_features= 512),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(0.2),
                       torch.nn.Linear(in_features=512, out_features= 128),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(0.2),
                       torch.nn.Linear(in_features=128, out_features= 2)
                       )

      def forward(self,image,CNV,meth,RNA):
         image_avg = self.resnet50_avg(image)
         image_avg = torch.flatten(image_avg,1,3)
         x = torch.cat([image_avg,CNV,meth,RNA],dim = 1)
         return self.MLP_mo(x)        
 

class resnet50_mo_print(torch.nn.Module):
      def __init__(self,dim_moinput = 5000):
         super(resnet50_mo_print, self).__init__()
         #resnet50_whole = models.resnet50(pretrained=True)
         #self.resnet50_avg = torch.nn.Sequential(*list(resnet50_whole.children())[:-1])
         self.MLP_mo = torch.nn.Sequential(
                       torch.nn.Linear(in_features=dim_moinput, out_features= 1024),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(0.2),
                       torch.nn.Linear(in_features=1024, out_features= 512),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(0.2),
                       torch.nn.Linear(in_features=512, out_features= 128),
                       torch.nn.ReLU(),
                       torch.nn.Dropout(0.2),
                       torch.nn.Linear(in_features=128, out_features= 2)
                       )

      def forward(self,image,CNV,meth,RNA):
         #image_avg = self.resnet50_avg(image)
         #image_avg = torch.flatten(image_avg,1,3)
         x = torch.cat([CNV,meth,RNA],dim = 1)
         # print(x.size())
         return self.MLP_mo(x)
