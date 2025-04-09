from sklearn.datasets import load_iris, load_wine, load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_auc_score, f1_score, hamming_loss
from sklearn.mixture import GaussianMixture

class RGM():
    def __init__(self, data, target, cluster_num):
        # data shape must be DxN
        self.X = data.copy()
        self.normalized_X = np.zeros_like(data)
        self.D = data.shape[0]
        self.N = data.shape[1]
        self.K = cluster_num
        self.target = target.copy()
        
        if self.D > self.N:
            self.small = self.N
        else:
            self.small = self.D

        self.G = np.zeros([self.N, self.small])
        self.Nk = np.zeros([self.small])
        self.ck = np.zeros([self.small])
        self.Mu = np.zeros([self.D, self.small])
        self.Sigma = np.zeros([self.D, self.D, self.small])
        self.w = 0

        self.gamma = np.zeros([self.K, self.N])
        self.proba = np.zeros([self.K, self.N])
        self.pred_label = np.zeros([self.N])
        self.pred_onehot_label = np.zeros([self.K, self.N])
        self.target_onehot_label = np.zeros([self.K, self.N])

        self.sklearn_proba = np.zeros([self.K, self.N])
        self.sklearn_pred_label = np.zeros([self.N])
        self.sklearn_pred_onehot_label = np.zeros([self.K, self.N])
        self.sklearn_target_onehot_label = np.zeros([self.K, self.N])
        
        self.auc_mean = 0
        self.auc_std = 0
        self.hl_mean = 0
        self.hl_std = 0
        self.f1_mean = 0
        self.f1_std = 0
        
        self.contour = np.zeros([self.K,50*50])
        x = np.linspace(-2.5,2.5,50)
        y = np.linspace(-2.5,2.5,50)
        self.xx ,self.yy = np.meshgrid(x,y)
        
    def fit(self):
        # normalize
        for i in range(self.D):
            if np.std(self.X[i,:]) > 0:
                self.normalized_X[i,:] = (self.X[i,:] - np.mean(self.X[i,:])) / (np.std(self.X[i,:]))
            else:
                self.normalized_X[i,:] = self.X[i,:] - np.mean(self.X[i,:])

        # eigenvalue decomposition or SVD
        U, s, VT = np.linalg.svd(self.normalized_X, full_matrices=False)

        # calculate G
        self.G[:,:] = VT.T
        for i in range(self.N):
            self.G[i,:] /= np.sqrt(np.sum(self.G[i,:] * self.G[i,:]))

        # calculate Nk
        self.Nk = (np.ones([self.N]).T@(self.G*self.G)).T

        # calculate ck
        for i in range(self.small):
            self.ck[i] = self.Nk[i] / self.N

        # calculate Mu
        for i in range(self.small):
            self.Mu[:,i] = 1/self.Nk[i] * self.normalized_X @ (self.G[:,i]*self.G[:,i])

        # calculate Sigma
        Sigma = np.zeros([self.D,self.D,self.small])
        for i in range(self.small):
            tmp = np.zeros([self.D,self.D])
            for n in range(self.N):
                tmp += 1/self.N * self.G[n,i]*self.G[n,i]*((self.normalized_X[:,n:n+1]-self.Mu[:,i]) @ \
                                                           (self.normalized_X[:,n:n+1]-self.Mu[:,i]).T)
            self.Sigma[:,:,i] = tmp

        # calculate w
        MuSig = self.Mu @ self.Mu.T / (self.small)
        XSig = (self.normalized_X) @ (self.normalized_X).T / (self.N)
        self.w = np.sqrt(np.sum(XSig*XSig)/np.sum(MuSig*MuSig))

        # modify Mu
        self.Mu = self.Mu * np.sqrt(self.w)
        
        # modify Sigma
        for i in range(self.small):
            tmp = np.zeros([self.D,self.D])
            for n in range(self.N):
                tmp += 1/self.N * self.G[n,i]*self.G[n,i]*((self.normalized_X[:,n:n+1]-self.Mu[:,i]) @ \
                                                           (self.normalized_X[:,n:n+1]-self.Mu[:,i]).T)
            self.Sigma[:,:,i] = tmp / self.w

    def PCAfit(self,reduce):
        # normalize
        for i in range(self.D):
            if np.std(self.X[i,:]) > 0:
                self.normalized_X[i,:] = (self.X[i,:] - np.mean(self.X[i,:])) / (np.std(self.X[i,:]))
            else:
                self.normalized_X[i,:] = self.X[i,:] - np.mean(self.X[i,:])

        U, s, VT = np.linalg.svd(self.normalized_X, full_matrices=False)
        self.normalized_X = (np.diag(s) @ VT)[:reduce,:]
        self.G = np.zeros([self.N, reduce])
        self.Nk = np.zeros([reduce])
        self.ck = np.zeros([reduce])
        self.Mu = np.zeros([reduce, reduce])
        self.Sigma = np.zeros([reduce, reduce, reduce])
        # eigenvalue decomposition or SVD
        U, s, VT = np.linalg.svd(self.normalized_X, full_matrices=False)

        # calculate G
        self.G[:,:] = VT.T
        for i in range(self.N):
            self.G[i,:] /= np.sqrt(np.sum(self.G[i,:] * self.G[i,:]))

        # calculate Nk
        self.Nk = (np.ones([self.N]).T@(self.G*self.G)).T

        # calculate ck
        for i in range(reduce):
            self.ck[i] = self.Nk[i] / self.N

        # calculate Mu
        for i in range(reduce):
            self.Mu[:,i] = 1/self.Nk[i] * self.normalized_X @ (self.G[:,i]*self.G[:,i])

        # calculate Sigma
        Sigma = np.zeros([reduce,reduce,reduce])
        for i in range(reduce):
            tmp = np.zeros([reduce,reduce])
            for n in range(self.N):
                tmp += 1/self.N * self.G[n,i]*self.G[n,i]*((self.normalized_X[:,n:n+1]-self.Mu[:,i]) @ \
                                                           (self.normalized_X[:,n:n+1]-self.Mu[:,i]).T)
            self.Sigma[:,:,i] = tmp

        # calculate w
        MuSig = self.Mu @ self.Mu.T / (self.D)
        XSig = (self.normalized_X) @ (self.normalized_X).T / (self.N)
        self.w = np.sqrt(np.sum(XSig*XSig)/np.sum(MuSig*MuSig))

        # modify Mu
        self.Mu = self.Mu * np.sqrt(self.w)
        
        # modify Sigma
        for i in range(reduce):
            tmp = np.zeros([reduce,reduce])
            for n in range(self.N):
                tmp += 1/self.N * self.G[n,i]*self.G[n,i]*((self.normalized_X[:,n:n+1]-self.Mu[:,i]) @ \
                                                           (self.normalized_X[:,n:n+1]-self.Mu[:,i]).T)
            self.Sigma[:,:,i] = tmp / self.w
        
        # select the separable Mu   
        dist = np.zeros([reduce,reduce])
        for i in range(reduce):
            for j in range(reduce):
                dist[i,j] = np.sum((self.Mu[:,i] - self.Mu[:,j])**2)
        without = []
        Mulist = []
        for k in range(self.K):
            ma = np.zeros([dist.shape[0]])
            for i in range(dist.shape[0]):
                if i not in without:
                    ma[i] = np.max(dist[i,:])
            Mulist.append(np.argmax(ma))
            without.append(np.argmax(ma))
        self.Mu = self.Mu[:,Mulist]
        self.Sigma = self.Sigma[:,:,Mulist]
        self.ck = self.ck[Mulist]
        
    def makegamma(self):
        ck2 = self.ck[:self.K]
        ck2 /= np.sum(ck2)
        Mu2 = self.Mu[:,:self.K]
        Sigma2 = self.Sigma[:,:,:self.K]
        for k in range(self.K):
            # avoid singular matrix
            det = np.linalg.det(Sigma2[:,:,k]+0.0001*np.eye(Sigma2.shape[0]))
            sinv = np.linalg.pinv(Sigma2[:,:,k]+0.0001*np.eye(Sigma2.shape[0]))
            
            for n in range(self.N):
                self.gamma[k,n] = ck2[k]*1/(2*np.pi)**(4/2.0)*1/(det)**(0.5)*np.exp( \
                    -1/2*((self.normalized_X[:,n]-Mu2[:,k]).T@sinv@(self.normalized_X[:,n]-Mu2[:,k])))
        
        for n in range(self.N):
            if np.sum(self.gamma[:,n]) > 0:
                self.gamma[:,n] /= np.sum(self.gamma[:,n])
            self.pred_label[n] = np.argmax(self.gamma[:,n])

    def maketargetonehot(self):
        pred_class = np.zeros([self.K])
        for k in range(self.K):
            kth_target = self.target[self.pred_label == k]
            # count[k] == the number of what is predicted as label k.
            # But the labels of the original target may be defferent from predicted labels \
            # because clustering does not know labels.
            # Using majority vote, predicted labels are modified to original labels.
            count = np.zeros([self.K])
            for i in range(kth_target.size):
                count[int(kth_target[i])] += 1
            pred_class[k] = np.argmax(count)    
        #for i in range(pred.size):
        #    if pred[i] >= Dreduced:
        #        pred[i] = np.random.randint(0,Dreduced-1)
        pred_target = np.zeros([self.N])
        for i in range(self.N):
            pred_target[i] = pred_class[int(self.pred_label[i])]
        for i in range(self.N):
            self.target_onehot_label[int(self.target[i]),i] = 1
            self.pred_onehot_label[int(pred_target[i]),i] = 1
    
    def maketargetproba(self):
        pred_class = np.zeros([self.K])
        for k in range(self.K):
            kth_target = self.target[self.pred_label == k]
            count = np.zeros([self.K])
            for i in range(kth_target.size):
                count[int(kth_target[i])] += 1
            pred_class[k] = np.argmax(count)
        #for i in range(pred.size):
        #    if pred[i] >= Dreduced:
        #        pred[i] = np.random.randint(0,Dreduced-1)
        pred_target = np.zeros([self.N])
        for i in range(self.N):
            pred_target[i] = pred_class[int(self.pred_label[i])]
        for i in range(self.N):
            self.target_onehot_label[int(self.target[i]),i] = 1
        
        for k in range(self.K):
            self.proba[k,:] = self.gamma[int(pred_class[k]),:]

    def sklearn_maketargetonehot(self):
        pred_class = np.zeros([self.K])
        for k in range(self.K):
            kth_target = self.target[self.sklearn_pred_label == k]
            # count[k] == the number of what is predicted as label k.
            # But the labels of the original target may be defferent from predicted labels \
            # because clustering does not know labels.
            # Using majority vote, predicted labels are modified to original labels.
            count = np.zeros([self.K])
            for i in range(kth_target.size):
                count[int(kth_target[i])] += 1
            pred_class[k] = np.argmax(count)    
        #for i in range(pred.size):
        #    if pred[i] >= Dreduced:
        #        pred[i] = np.random.randint(0,Dreduced-1)
        pred_target = np.zeros([self.N])
        for i in range(self.N):
            pred_target[i] = pred_class[int(self.sklearn_pred_label[i])]
        for i in range(self.N):
            self.sklearn_target_onehot_label[int(self.target[i]),i] = 1
            self.sklearn_pred_onehot_label[int(pred_target[i]),i] = 1
    def sklearn_maketargetproba(self):
        pred_class = np.zeros([self.K])
        for k in range(self.K):
            kth_target = self.target[self.sklearn_pred_label == k]
            count = np.zeros([self.K])
            for i in range(kth_target.size):
                count[int(kth_target[i])] += 1
            pred_class[k] = np.argmax(count)
        #for i in range(pred.size):
        #    if pred[i] >= Dreduced:
        #        pred[i] = np.random.randint(0,Dreduced-1)
        pred_target = np.zeros([self.N])
        for i in range(self.N):
            pred_target[i] = pred_class[int(self.sklearn_pred_label[i])]
        for i in range(self.N):
            self.target_onehot_label[int(self.target[i]),i] = 1

        tmp = np.zeros_like(self.sklearn_proba)
        for k in range(self.K):
            tmp[k,:] = self.sklearn_proba[int(pred_class[k]),:]
        self.sklearn_proba[:,:] = tmp
        
    def sklearn_GMM(self):
        auc = np.zeros([1000])
        hl = np.zeros([1000])
        f1 = np.zeros([1000])
        for i in range(1000):
            clf = GaussianMixture(n_components=self.K, init_params='random')
            clf.fit(self.normalized_X.T)
            self.sklearn_pred_label = clf.predict(self.normalized_X.T).T
            self.sklearn_proba = clf.predict_proba(self.normalized_X.T).T
            self.sklearn_maketargetonehot()
            self.sklearn_maketargetproba()
            auc[i] = roc_auc_score(self.sklearn_target_onehot_label.T,self.sklearn_proba.T,  average='micro')
            hl[i] = hamming_loss(self.sklearn_target_onehot_label.T,self.sklearn_pred_onehot_label.T)
            f1[i] = f1_score(self.sklearn_target_onehot_label.T,self.sklearn_pred_onehot_label.T,  average='micro')
        self.auc_mean = np.mean(auc)
        self.auc_std = np.std(auc)
        self.hl_mean = np.mean(hl)
        self.hl_std = np.std(hl)
        self.f1_mean = np.mean(f1)
        self.f1_std = np.std(f1)
        
    def makeiriscontour(self):
        xxx = np.zeros([self.D,50*50])
        xxx[0,:] = self.xx.flatten()
        xxx[1,:] = self.yy.flatten()
        
        for i in range(self.K):
            det = np.linalg.det(self.Sigma[:,:,i])
            #print(det)
            sinv = np.linalg.inv(self.Sigma[:,:,i])
            sinv = sinv[:2,:2]
            for n in range(50*50):
                self.contour[i,n] = self.ck[i]*1/(2*np.pi)**(4/2.0)*1/(det)**(0.5)*np.exp( \
                    -1/2*((xxx[:2,n]-self.Mu[:2,i]).T@sinv@(xxx[:2,n]-self.Mu[:2,i])))

    def makeirisimage(self):
        plt.rcParams["font.size"] = 14
        plt.rcParams["font.family"] = "Times New Roman" 
        fig, ax = plt.subplots(1,1,figsize=(4,4))
        cols = ['red','blue','green','cyan']
        
        for i in range(self.K):
            cs =ax.contour(self.xx[:,:],self.yy[:,:],self.contour[i,:].reshape([50,50])[:,:], levels=3,colors='black', linewidths=1, extend='neither')
            #ax.clabel(cs)
        ax.scatter(self.normalized_X[0,:50], self.normalized_X[1,:50], s=10, color='red', label='setosa')
        ax.scatter(self.normalized_X[0,50:100], self.normalized_X[1,50:100], s=10, color='blue', label='versicolour')
        ax.scatter(self.normalized_X[0,100:150], self.normalized_X[1,100:150], s=10, color='green', label='virginica')
        
        for i in range(self.K):
            ax.scatter(self.Mu[0,i], self.Mu[1,i],marker='x',s=100, color='black')
        plt.xlabel("normalized sepal length")
        plt.ylabel("normalized sepal width")
        plt.legend()
        csize = 2.5
        plt.xlim(-csize,csize)
        plt.ylim(-csize,csize)
        
        plt.tight_layout()
        plt.savefig('obaya1.svg')
        plt.show()

if __name__ == '__main__':
    iris = load_iris()
    iris_data = iris.data
    iris_target = iris.target
    iris_target_names = iris.target_names
    
    wine = load_wine()
    wine_data = wine.data
    wine_target = wine.target
    wine_target_names = wine.target_names
    
    digits = load_digits()
    digits_data = digits.data
    digits_target = digits.target
    digits_target_names = digits.target_names
    
    print('iris')
    print('    ... RGM calculating ...')
    cluster_num = 3
    iris_rgm = RGM(iris_data.T, iris_target, cluster_num)
    iris_rgm.fit()
    iris_rgm.makegamma()
    iris_rgm.maketargetonehot()
    iris_rgm.maketargetproba()
    iris_rgm.makeiriscontour()
    #hoge.makeirisimage()
    print('    ... sklearn calculating ... it takes long time ...')
    iris_rgm.sklearn_GMM()
    
    print('    proposed method:')
    print('        ROC AUC Score:','{:.3g}'.format(roc_auc_score(iris_rgm.target_onehot_label.T,iris_rgm.proba.T,  average='micro')))
    print('              F-score:','{:.3g}'.format(f1_score(iris_rgm.target_onehot_label.T,iris_rgm.pred_onehot_label.T,  average='micro')))
    print('         hamming loss:','{:.3g}'.format(hamming_loss(iris_rgm.target_onehot_label.T,iris_rgm.pred_onehot_label.T)))

    print('    sklearn:')
    print('        ROC AUC Score: mean...','{:.3g}'.format(iris_rgm.auc_mean), 'std...', '{:.3g}'.format(iris_rgm.auc_std))
    print('              F-score: mean...', '{:.3g}'.format(iris_rgm.f1_mean), 'std...', '{:.3g}'.format(iris_rgm.f1_std) )
    print('         hamming loss: mean...','{:.3g}'.format(iris_rgm.hl_mean), 'std...', '{:.3g}'.format(iris_rgm.hl_std))

    print('wine')
    print('    ... RGM calculating ...')
    cluster_num = 3
    wine_rgm = RGM(wine_data.T, wine_target, cluster_num)
    wine_rgm.fit()
    wine_rgm.makegamma()
    wine_rgm.maketargetonehot()
    wine_rgm.maketargetproba()
    print('    ... sklearn calculating ... it takes long time ...')
    wine_rgm.sklearn_GMM()

    print('    proposed method:')
    print('        ROC AUC Score:','{:.3g}'.format(roc_auc_score(wine_rgm.target_onehot_label.T,wine_rgm.proba.T,  average='micro')))
    print('              F-score:','{:.3g}'.format(f1_score(wine_rgm.target_onehot_label.T,wine_rgm.pred_onehot_label.T,  average='micro')))
    print('         hamming loss:','{:.3g}'.format(hamming_loss(wine_rgm.target_onehot_label.T,wine_rgm.pred_onehot_label.T)))

    print('    sklearn:')
    print('        ROC AUC Score: mean...','{:.3g}'.format(wine_rgm.auc_mean), 'std...', '{:.3g}'.format(wine_rgm.auc_std))
    print('              F-score: mean...', '{:.3g}'.format(wine_rgm.f1_mean), 'std...', '{:.3g}'.format(wine_rgm.f1_std) )
    print('         hamming loss: mean...','{:.3g}'.format(wine_rgm.hl_mean), 'std...', '{:.3g}'.format(wine_rgm.hl_std))

    print('digits')
    print('    ... RGM calculating ...')
    cluster_num = 10
    digits_rgm = RGM(digits_data.T, digits_target, cluster_num)
    digits_rgm.fit()
    digits_rgm.makegamma()
    digits_rgm.maketargetonehot()
    digits_rgm.maketargetproba()
    print('    ... sklearn calculating ... it takes long time ...')
    digits_rgm.sklearn_GMM()

    print('    proposed method:')
    print('        ROC AUC Score:','{:.3g}'.format(roc_auc_score(digits_rgm.target_onehot_label.T,digits_rgm.proba.T,  average='micro')))
    print('              F-score:','{:.3g}'.format(f1_score(digits_rgm.target_onehot_label.T,digits_rgm.pred_onehot_label.T,  average='micro')))
    print('         hamming loss:','{:.3g}'.format(hamming_loss(digits_rgm.target_onehot_label.T,digits_rgm.pred_onehot_label.T)))

    print('    sklearn:')
    print('        ROC AUC Score: mean...','{:.3g}'.format(digits_rgm.auc_mean), 'std...', '{:.3g}'.format(digits_rgm.auc_std))
    print('              F-score: mean...', '{:.3g}'.format(digits_rgm.f1_mean), 'std...', '{:.3g}'.format(digits_rgm.f1_std) )
    print('         hamming loss: mean...','{:.3g}'.format(digits_rgm.hl_mean), 'std...', '{:.3g}'.format(digits_rgm.hl_std))

    print('with PCA')
    print('iris')
    print('    ... RGM calculating ...')
    cluster_num = 3
    iris_rgm = RGM(iris_data.T, iris_target, cluster_num)
    iris_rgm.PCAfit(4)
    iris_rgm.makegamma()
    iris_rgm.maketargetonehot()
    iris_rgm.maketargetproba()
    iris_rgm.makeiriscontour()
    #hoge.makeirisimage()
    print('    ... sklearn calculating ... it takes long time ...')
    iris_rgm.sklearn_GMM()
    
    print('    proposed method:')
    print('        ROC AUC Score:','{:.3g}'.format(roc_auc_score(iris_rgm.target_onehot_label.T,iris_rgm.proba.T,  average='micro')))
    print('              F-score:','{:.3g}'.format(f1_score(iris_rgm.target_onehot_label.T,iris_rgm.pred_onehot_label.T,  average='micro')))
    print('         hamming loss:','{:.3g}'.format(hamming_loss(iris_rgm.target_onehot_label.T,iris_rgm.pred_onehot_label.T)))

    print('    sklearn:')
    print('        ROC AUC Score: mean...','{:.3g}'.format(iris_rgm.auc_mean), 'std...', '{:.3g}'.format(iris_rgm.auc_std))
    print('              F-score: mean...', '{:.3g}'.format(iris_rgm.f1_mean), 'std...', '{:.3g}'.format(iris_rgm.f1_std) )
    print('         hamming loss: mean...','{:.3g}'.format(iris_rgm.hl_mean), 'std...', '{:.3g}'.format(iris_rgm.hl_std))

    print('wine')
    print('    ... RGM calculating ...')
    cluster_num = 3
    wine_rgm = RGM(wine_data.T, wine_target, cluster_num)
    wine_rgm.PCAfit(5)
    wine_rgm.makegamma()
    wine_rgm.maketargetonehot()
    wine_rgm.maketargetproba()
    print('    ... sklearn calculating ... it takes long time ...')
    wine_rgm.sklearn_GMM()

    print('    proposed method:')
    print('        ROC AUC Score:','{:.3g}'.format(roc_auc_score(wine_rgm.target_onehot_label.T,wine_rgm.proba.T,  average='micro')))
    print('              F-score:','{:.3g}'.format(f1_score(wine_rgm.target_onehot_label.T,wine_rgm.pred_onehot_label.T,  average='micro')))
    print('         hamming loss:','{:.3g}'.format(hamming_loss(wine_rgm.target_onehot_label.T,wine_rgm.pred_onehot_label.T)))

    print('    sklearn:')
    print('        ROC AUC Score: mean...','{:.3g}'.format(wine_rgm.auc_mean), 'std...', '{:.3g}'.format(wine_rgm.auc_std))
    print('              F-score: mean...', '{:.3g}'.format(wine_rgm.f1_mean), 'std...', '{:.3g}'.format(wine_rgm.f1_std) )
    print('         hamming loss: mean...','{:.3g}'.format(wine_rgm.hl_mean), 'std...', '{:.3g}'.format(wine_rgm.hl_std))

    print('digits')
    print('    ... RGM calculating ...')
    cluster_num = 10
    digits_rgm = RGM(digits_data.T, digits_target, cluster_num)
    digits_rgm.PCAfit(64)
    digits_rgm.makegamma()
    digits_rgm.maketargetonehot()
    digits_rgm.maketargetproba()
    print('    ... sklearn calculating ... it takes long time ...')
    digits_rgm.sklearn_GMM()

    print('    proposed method:')
    print('        ROC AUC Score:','{:.3g}'.format(roc_auc_score(digits_rgm.target_onehot_label.T,digits_rgm.proba.T,  average='micro')))
    print('              F-score:','{:.3g}'.format(f1_score(digits_rgm.target_onehot_label.T,digits_rgm.pred_onehot_label.T,  average='micro')))
    print('         hamming loss:','{:.3g}'.format(hamming_loss(digits_rgm.target_onehot_label.T,digits_rgm.pred_onehot_label.T)))

    print('    sklearn:')
    print('        ROC AUC Score: mean...','{:.3g}'.format(digits_rgm.auc_mean), 'std...', '{:.3g}'.format(digits_rgm.auc_std))
    print('              F-score: mean...', '{:.3g}'.format(digits_rgm.f1_mean), 'std...', '{:.3g}'.format(digits_rgm.f1_std) )
    print('         hamming loss: mean...','{:.3g}'.format(digits_rgm.hl_mean), 'std...', '{:.3g}'.format(digits_rgm.hl_std))
