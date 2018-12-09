# 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GRU
from keras.layers import Flatten, Dropout, Dense, Embedding
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras.callbacks import BaseLogger
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from matplotlib.colors import BoundaryNorm
import pandas as pd
import numpy as np
import random
from time import time
import time
from sklearn.manifold import TSNE
from sklearn import decomposition
import itertools
from scipy.cluster.hierarchy import linkage
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import dendrogram


#%% FUNCTION FOR TRAINING THE MODEL 

def lstmtrain(X_train,Y_train,X_test,Y_test):
    
    
    # Compile and train different models while measuring performance.
    results = []
    max_features = X.max()+1
    
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim,mask_zero=True)) # , input_length=SEQLEN 
    
    #if you would lke to add an additional convolutional layer
    #model.add(Conv1D(filters=embedding_dim, kernel_size=3, padding='same', activation='relu'))
    #model.add(MaxPooling1D()) #pool_size=Full)
    
    #model.add(Dropout(0.05))
    
    model.add(LSTM(nUNIT)) #   GRU dropout=0.01, recurrent_dropout=0.01,

 #  model.add(LSTM(nUNIT,  implementation=1)) #dropout=0.01, recurrent_dropout=0.01,

    model.add(Dense(Y_train.shape[1], activation='softmax')) #
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    #tensorboard = TensorBoard(
    #    log_dir="./tensorboard/",
    #    write_images=True, 
    #    histogram_freq=1,
    #    embeddings_freq=250, 
    #    embeddings_metadata='map.tsv'
    #)
    #tensorboard --logdir=C:\Users\resea\Dropbox\deep_chem\Modellek\FINAL
    #http://fmt_otka:6006
    print(model.summary())
    
    start_time = time.time()
    history = model.fit( X_train,  Y_train,  validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs)
    average_time_per_epoch = (time.time() - start_time) / epochs
    results.append((history, average_time_per_epoch))
    
        
    return(model,results)


def stackedtrain(X_train,Y_train,X_test,Y_test):


    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim, input_length=SEQLEN,mask_zero=True))
    model.add(LSTM(nUNIT, return_sequences=True,
                   input_shape=(SEQLEN, embedding_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(nUNIT, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(nUNIT))  # return a single vector of dimension 32
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', #rmsprop
                  metrics=['accuracy'])
    
    print(model.summary())
    
    start_time = time.time()
    history = model.fit( X_train,  Y_train,  validation_data=(X_test, Y_test), batch_size=X.shape[0], epochs=epochs)
    average_time_per_epoch = (time.time() - start_time) / epochs
    results.append((history, average_time_per_epoch))
    

    return(model,results)

#%% FUNCTION FOR THE X-FOLD CROSS VALDATION 

def crossval(X,Y,n_fold):
    seed = 42
    np.random.seed(seed)
    K_fold = StratifiedKFold(n_fold, shuffle=False, random_state=seed)
    cv_scores=[]
    cv_models = []
    for train, test in K_fold.split(X,y_label): #numerikus label kell 
        (model,results)=lstmtrain(X[train],  Y[train], X[test],  Y[test])
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("Model Accuracy: %.2f%%" % (scores[1]*100))
        cv_scores.append(scores[1] * 100)
        cv_models.append(model)
    return(cv_models, cv_scores)

#%% FUNCTION FOR LOADING THE DATA

def loadandprepare(filename,withR,nEVENT):
    
    print('Loading data...')
    X_data = pd.read_excel(filename, header=None, dtype=float)
    y_data = pd.read_excel('Fault_0822.xlsx', header=None, dtype=int)
    X = X_data.as_matrix()
    Y = y_data.as_matrix()
    
    if revFlag:
        X=np.flip(X,1)

    if withR:
        SEQLEN=nEVENT*2-1
    else:
        SEQLEN=nEVENT
    
    if not(withR):
        X=X[:,::2] 

    # SEQUENCE generation 
    if revFlag:
        X = sequence.pad_sequences(X, maxlen=SEQLEN, dtype='int32', padding='pre', truncating='pre', value=-5.)
    else:
        X = sequence.pad_sequences(X, maxlen=SEQLEN, dtype='int32', padding='post', truncating='post', value=-5.)
        

    shape_X = X.shape    
    shape_Y = Y.shape    
    
    # CODING for ONE- HOT ENCODING (and decoding transf)
    XEncoder = preprocessing.LabelEncoder()
    XEncoder.fit(X.flatten()) 
    X_encoded = XEncoder.transform(X.flatten()) #inverse_transform-al majd vissza
    X=X_encoded.reshape((shape_X))
    
    YEncoder = preprocessing.LabelEncoder()
    YEncoder.fit(Y.flatten()) 
    Y_encoded = YEncoder.fit_transform(Y)
    y_label=Y_encoded
    Y=Y_encoded.reshape((shape_Y)) 
    
    Y = np_utils.to_categorical(Y)
    
    
    return(X,Y,SEQLEN,XEncoder,YEncoder,y_label)



#%%  MAIN PARAMETERS


filenames=['Sequences_simple_alarm_0918.xlsx', 'Sequences_alarm_warning_0918.xlsx', 'Sequences_quantized_0918.xlsx']
dataname=['A','B','C']




n_fold = 7
epochs = 500
batch_size = 512





#%% ANALYZIS OF the effect of the training data 

nEVENT=5
revFlag=False 
nUNIT=11
embedding_dim = 4


withR=True

acc_data1 = []
for di in range(3):
    filename=filenames[di]
    (X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,nEVENT)
    (m1,cv_scores)=crossval(X,Y,n_fold)
    acc_data1.append(cv_scores)


fig = plt.figure()
ax = fig.add_subplot(111)
bp = ax.boxplot(acc_data1)
plt.ylabel('Correct Classification Rate (%)')
ax.set_xticklabels(['A/1','B/1','C/1'])
plt.xlabel('Datasets')
plt.savefig('acc_data_ccr.png', dpi=300)
plt.show()

ad1 = pd.DataFrame(acc_data1)
writer = pd.ExcelWriter('acc_data')
ad1.to_excel(writer,'with R')
##%% ANALYZIS OF the effect of the number of events and the temporal relationship

nUNIT=11
revFlag=False 



di=1 #0-2
filename=filenames[di]


embedding_dim=4

VnEVENT=[2, 3, 4, 5, 6]

casename=['with R', 'without R']
case=[True, False]


withR=True
acc_R = []
for di in range(len(VnEVENT)):
    nEVENT=VnEVENT[di]
    (X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,nEVENT)
    (m2,cv_scores)=crossval(X,Y,n_fold)
    acc_R.append(cv_scores)

withR=False
acc_Rwo = []
for di in range(len(VnEVENT)):
    nEVENT=VnEVENT[di]
    (X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,nEVENT)
    (m2,cv_scores)=crossval(X,Y,n_fold)
    acc_Rwo.append(cv_scores)

fig = plt.figure() 
ax = fig.add_subplot(211)
bp = ax.boxplot(acc_R)
plt.ylabel('Corr. Class. Rate (%)')
plt.title('With R',Fontsize=8)
#ax.set_xticklabels(VnEVENT)
ax.set_xticklabels([])
plt.ylim((87.5,95))

ax = fig.add_subplot(212)
plt.title('Without R',Fontsize=8)
bp = ax.boxplot(acc_Rwo)
plt.ylabel('Corr. Class. Rate (%)')
ax.set_xticklabels(VnEVENT)
plt.xlabel('# of Events')
plt.ylim((87.5,95))

plt.savefig('R_type2.png', dpi=300)
plt.show()
#
adR1 = pd.DataFrame(acc_R)
adR2 = pd.DataFrame(acc_Rwo)

writer = pd.ExcelWriter('acc_ER2.xlsx')
adR1.to_excel(writer,'R')
adR2.to_excel(writer,'Rwo')
writer.save()    


dims = [2,3,4,6]  #embedding_dim

acc_emb = []
for embedding_dim in dims:
    (m3,cv_scores) = crossval(X,Y,n_fold)
    acc_emb.append(cv_scores)
    
fig = plt.figure()
ax = fig.add_subplot(111)
bp = ax.boxplot(acc_emb)
#plt.title('Effect of embedding dimension on accuracy', fontsize=8)
plt.ylabel('Correct Classification Rate (%)')
plt.xlabel('Embedding Dimension')
ax.set_xticklabels(dims)
plt.savefig('nEsmb_crossval_proba.png', dpi=300)
plt.show()

#%% confusion matrix


seed = 42
np.random.seed(seed)
  
di=1
filename=filenames[di]

revFlag=False
nEVENT=4
nUNIT=11
embedding_dim = 4
withR=False


(X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,nEVENT)


(model,results)=lstmtrain(X,Y,X,Y)
scores = model.evaluate(X, Y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

predictions=model.predict(X, batch_size=batch_size)

Confusion=confusion_matrix(np.argmax(Y,axis=1), np.argmax(predictions,axis=1))

#categories = list(set(YEncoder.inverse_transform(y_label)))
categories = list(set(y_label+1))

tick_marks = np.arange(len(categories))
plt.figure(figsize=(11,11))
plt.imshow(Confusion, cmap=plt.cm.Blues)
plt.xticks(tick_marks,categories)
plt.yticks(tick_marks,categories)
plt.title('')
plt.xlabel('Predicted class', fontsize=20)
plt.ylabel('True class', fontsize=20)
for i, j in itertools.product(range(Confusion.shape[0]), range(Confusion.shape[1])):
    c='black'
    if Confusion[i, j]/200 > .50:
        c='white'
    plt.text(j, i, format(Confusion[i, j], '.0f'), horizontalalignment="center",color=c)

plt.savefig('Confmatabszolut.png', dpi=300)
plt.show()



#%% Visualization of the model 
def get_activations(model, layer, X):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X,0])
    return activations

EACT = get_activations(model,0,X)
EACT=np.array(EACT[0])

plt.yticks(range(0,EACT.shape[0],200))
plt.imshow( np.reshape(EACT,(EACT.shape[0],SEQLEN*embedding_dim)), aspect='auto',cmap='coolwarm', clim=(EACT.min(), EACT.max()))
plt.colorbar()
plt.xlabel('# LSTM node')
plt.ylabel('# sequence')
plt.savefig('Layer0.png', dpi=300)
plt.show()
LACT = get_activations(model,1,X)
LACT=np.array(LACT[0])
plt.imshow(LACT, aspect='auto',cmap=plt.cm.seismic, clim=(-1,1))
plt.title('LSTM unit activation')
plt.xlabel('unit')
plt.ylabel('sample')
plt.xticks(range(0,nUNIT,10))
plt.yticks(range(0,LACT.shape[0],200))
plt.colorbar()
plt.savefig('lstm_activation.png', dpi=300)
plt.show()
#
CACT = get_activations(model,2,X)
CACT=np.array(CACT[0])
plt.title('Output layer activation')
plt.xlabel('categories')
plt.ylabel('samples')
plt.xticks(range(0,len(categories)+1),categories, rotation='vertical')
plt.yticks(range(0,LACT.shape[0],200))
plt.imshow(CACT, aspect='auto',cmap='Oranges', clim=(0,1))
plt.colorbar()
plt.savefig('Layer3.png', dpi=300)
plt.show()
#
#

#
#%% PCA of Output layer

LACT = get_activations(model,1,X)
Lpca = decomposition.PCA()
Lpca.fit(LACT[0])
Lp = Lpca.transform(LACT[0])
variance = Lpca.explained_variance_ratio_

plt.figure(figsize=(11,11))
#plt.title('PCA of LSTM activation')
xplot_label = '$t_1$ ({}% of variance)'.format(int(variance[0]*100))
yplot_label = '$t_2$ ({}% of variance)'.format(int(variance[1]*100))
plt.xlabel(xplot_label, fontsize=20)
plt.ylabel(yplot_label, fontsize=20, font)
yhat=np.argmax(predictions,axis=1)+1
bounds = np.linspace(0,11,12)
cmap = plt.get_cmap('jet', 11)
norm = BoundaryNorm(bounds, cmap.N)
plt.scatter(Lp[:,0], Lp[:,1], cmap=cmap, c=yhat, norm=norm)
cbar = plt.colorbar(ticks=bounds, boundaries=bounds)
cbar.ax.get_yaxis().set_ticks([])
cats = list(set(YEncoder.inverse_transform(y_label)))
for cat in range(11):
    cbar.ax.text(1.5,np.linspace(0,1,12)[cat]+0.045,cat+1,ha='center', va='center')
plt.savefig('PCA_act_label.png', dpi=300)
plt.show()

plt.bar(np.arange(len(variance)),variance*100)
plt.plot(variance.cumsum()*100)
plt.xticks(np.arange(len(variance)),np.arange(1,len(variance)+1))
plt.xlabel('No. of eigenvalues')
plt.ylabel('Cumulated variance percentage (%)')
plt.savefig('PCA_var_act.png', dpi=300)
plt.show()

#%% Visualization of the parameters of the EMBEDDING layer 

We=model.layers[0].get_weights()
We=We[0];
V, D = We.shape
pca = decomposition.PCA()
pca.fit(We)
Wep = pca.transform(We)

labels=XEncoder.inverse_transform(range(V))

xvector = pca.components_[0]
yvector = pca.components_[1]

xs = Wep[:,0]
ys = Wep[:,1]

center = Wep.mean(axis=0)
cov = pca.get_covariance()[:2][:,0:2]
vals, vecs = np.linalg.eigh(cov)
order = vals.argsort()[::-1]
vals, vecs = vals[order], vecs[order]
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
width, height = 2 * 2 * np.sqrt(vals)

plt.figure(figsize=(11,11))
ax = plt.gca()
ax.grid(True)

plt.ylim([-Wep[:,1].max()-0.1, Wep[:,1].max()+0.1])
plt.xlim([-Wep[:,0].max()-0.1, Wep[:,0].max()+0.1])
plt.title('PCA of Embedding weights')
from matplotlib.patches import Ellipse
for i in range(len(xvector)):
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', length_includes_head=True, width=0.0005, head_width=0.01)
    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
             i, color='r')
plt.scatter(Wep[:,0], Wep[:,1])
for k in range(1,V):
    plt.text(Wep[k,0],Wep[k,1],labels[k])
variance = pca.explained_variance_ratio_
xplot_label = '$t_1$ ({}% of variance)'.format(int(variance[0]*100))
yplot_label = '$t_2$ ({}% of variance)'.format(int(variance[1]*100))
plt.xlabel(xplot_label)
plt.ylabel(yplot_label)
ax.add_patch(Ellipse(xy=center, width=width, height=height, angle=theta, edgecolor='k', facecolor='none'))
plt.savefig('pca_loadings.png', dpi=300)
plt.show()

#%% Variance analysis of the emb.


plt.bar(range(len(variance)),variance)
plt.plot(variance.cumsum())
plt.savefig('pca_var_loadings.png', dpi=300)
plt.show()


#%% PCA of the embedidng weighs
We=model.layers[0].get_weights()
We=We[0][1::];

D_We = pairwise_distances(We)

DD=D_We/D_We.max()
Z = linkage(DD, method='average')
dendrogram(Z, labels=labels)
plt.xlabel('Event ID')
plt.ylabel('Distance')

plt.savefig('den_pca_emb_ID_Distance.png', dpi=300)
#%%

from __future__ import division, print_function
import numpy as np

__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = "1.0.3"
__license__ = "MIT"


def hyperellipsoid(P, y=None, z=None, pvalue=.95, units=None, show=True, ax=None, covar=None, variance=None):
    """
    Prediction hyperellipsoid for multivariate data.

    The hyperellipsoid is a prediction interval for a sample of a multivariate
    random variable and is such that there is pvalue*100% of probability that a
    new observation will be contained inside the hyperellipsoid [1]_.  
    The hyperellipsoid is also a tolerance region such that the average or
    expected value of the proportion of the population contained in this region
    is exactly pvalue*100% (called Type 2 tolerance region by Chew (1966) [1]_).

    The directions and lengths of the semi-axes are found, respectively, as the
    eigenvectors and eigenvalues of the covariance matrix of the data using
    the concept of principal components analysis (PCA) [2]_ or singular value
    decomposition (SVD) [3]_ and the length of the semi-axes are adjusted to
    account for the necessary prediction probability.

    The volume of the hyperellipsoid is calculated with the same equation for
    the volume of a n-dimensional ball [4]_ with the radius replaced by the
    semi-axes of the hyperellipsoid.

    This function calculates the prediction hyperellipsoid for the data,
    which is considered a (finite) sample of a multivariate random variable
    with normal distribution (i.e., the F distribution is used and not
    the approximation by the chi-square distribution).

    Parameters
    ----------
    P : 1-D or 2-D array_like
        For a 1-D array, P is the abscissa values of the [x,y] or [x,y,z] data.
        For a 2-D array, P is the joined values of the multivariate data.
        The shape of the 2-D array should be (n, p) where n is the number of
        observations (rows) and p the number of dimensions (columns).
    y : 1-D array_like, optional (default = None)
        Ordinate values of the [x, y] or [x, y, z] data.
    z : 1-D array_like, optional (default = None)
        Ordinate values of the [x, y] or [x, y, z] data.
    pvalue : float, optional (default = .95)
        Desired prediction probability of the hyperellipsoid.
    units : str, optional (default = None)
        Units of the input data.
    show : bool, optional (default = True)
        True (1) plots data in a matplotlib figure, False (0) to not plot.
        Only the results for p=2 (ellipse) or p=3 (ellipsoid) will be plotted.
    ax : a matplotlib.axes.Axes instance (default = None)
"""

    from scipy.stats import f as F
    from scipy.special import gamma

    P = np.array(P, ndmin=2, dtype=float)
    if P.shape[0] == 1:
        P = P.T
    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=float)
        if y.shape[0] == 1:
            y = y.T
        P = np.concatenate((P, y), axis=1)
    if z is not None:
        z = np.array(z, copy=False, ndmin=2, dtype=float)
        if z.shape[0] == 1:
            z = z.T
        P = np.concatenate((P, z), axis=1)
    cov = covar
    U, s, Vt = np.linalg.svd(cov)
    p, n = s.size, P.shape[0]
    # F percent point function
    fppf = F.ppf(pvalue, p, n-p)*(n-1)*p*(n+1)/n/(n-p)
    # semi-axes (largest first)
    saxes = np.sqrt(s*fppf)
    hypervolume = np.pi**(p/2)/gamma(p/2+1)*np.prod(saxes)
    # rotation matrix
    if p == 2 or p == 3:
        R = Vt
        if s.size == 2:
            #angles = np.array([np.rad2deg(np.arctan2(R[1, 0], R[0, 0])), 90-np.rad2deg(np.arctan2(R[1, 0], -R[0, 0]))])
            angles = None
        else:
            # angles = rotXYZ(R, unit='deg')
            angles = None
        # centroid of the hyperellipsoid
        center = np.mean(P, axis=0)
    else:
        R, angles = None, None

    if show and (p == 2 or p == 3):
        _plot(P, hypervolume, saxes, center, R, pvalue, units, ax)

    return hypervolume, saxes, angles, center, R


def _plot(P, hypervolume, saxes, center, R, pvalue, units, ax):
    """Plot results of the hyperellipsoid function, see its help."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        # code based on https://github.com/minillinim/ellipsoid:
        # parametric equations
        u = np.linspace(0, 2*np.pi, 100)
        #if saxes.size == 2:
        x = saxes[0]*np.cos(u)
        y = saxes[1]*np.sin(u)
        # rotate data
#        for i in range(len(x)):
#            [x[i], y[i]] = np.dot([x[i], y[i]], R) + center
#    else:
        if saxes.size == 3:
            v = np.linspace(0, np.pi, 100)
            x = saxes[0]*np.outer(np.cos(u), np.sin(v))
            y = saxes[1]*np.outer(np.sin(u), np.sin(v))
            z = saxes[2]*np.outer(np.ones_like(u), np.cos(v))
            # rotate data
#            for i in range(len(x)):
#                for j in range(len(x)):
#                    [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], R) + center

        if saxes.size == 2:
            if ax is None:
                fig, ax = plt.subplots(1, 1)
            # plot raw data
            ax.plot(P[:, 0], P[:, 1], '.', color=[0, 0, 1, .5])
            # plot ellipse
            ax.plot(x, y, '--', color=[1, 0, 0, 1], linewidth=1.5)
            # plot axes
            for i in range(saxes.size):
                print(saxes)
                # rotate axes
                # a = np.dot(np.diag(saxes)[i], R).reshape(2, 1)
                # points for the axes extremities
                """
                a = np.dot(a, np.array([-1, 1], ndmin=2))+center.reshape(2, 1)
                ax.plot(a[0], a[1], color=[1, 0, 0, .6], linewidth=2)
                ax.text(a[0, 1], a[1, 1], '%d' % (i + 1),
                        fontsize=20, color='r')
            """
            plt.axis('equal')
            plt.grid()
            title = r'Prediction ellipse (p=%4.2f): Area=' % pvalue
            if units is not None:
                units2 = ' [%s]' % units
                units = units + r'$^2$'
                title = title + r'%.2f %s' % (hypervolume, units)
            else:
                units2 = ''
                title = title + r'%.2f' % hypervolume
        else:
            from mpl_toolkits.mplot3d import Axes3D
            if ax is None:
                fig = plt.figure()
                ax = fig.add_axes([0, 0, 1, 1], projection='3d')
            ax.view_init(20, 30)
            # plot raw data
            ax.plot(P[:, 0], P[:, 1], P[:, 2], '.', color=[0, 0, 1, .4])
            # plot ellipsoid
            ax.plot_surface(x, y, z, rstride=5, cstride=5, color=[0, 1, 0, .1],
                            linewidth=1, edgecolor=[.1, .9, .1, .4])
            # ax.plot_wireframe(x, y, z, color=[0, 1, 0, .5], linewidth=1)
            #                  rstride=3, cstride=3, edgecolor=[0, 1, 0, .5])
            # plot axes
            for i in range(saxes.size):
                # rotate axes
                # a = np.dot(np.diag(saxes)[i], R).reshape(3, 1)
                # points for the axes extremities
                """
                a = np.dot(a, np.array([-1, 1], ndmin=2))+center.reshape(3, 1)
                ax.plot(a[0], a[1], a[2], color=[1, 0, 0, .6], linewidth=2)
                ax.text(a[0, 1], a[1, 1], a[2, 1], '%d' % (i+1),
                        fontsize=20, color='r')
                """
            lims = [np.min([P.min(), x.min(), y.min(), z.min()]),
                    np.max([P.max(), x.max(), y.max(), z.max()])]
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_zlim(lims)
            title = r'Prediction ellipsoid (p=%4.2f): Volume=' % pvalue
            if units is not None:
                units2 = ' [%s]' % units
                units = units + r'$^3$'
                title = title + r'%.2f %s' % (hypervolume, units)
            else:
                units2 = ''
                title = title + r'%.2f' % hypervolume
            zplot_label = '$t_2$ ({}% of variance)'.format(int(variance[2]*100))
            ax.set_zlabel(zplot_label, fontsize=14)
            # plt.zticks(fontsize=12)
        
        xplot_label = '$t_1$ ({}% of variance)'.format(int(variance[0]*100))
        yplot_label = '$t_2$ ({}% of variance)'.format(int(variance[1]*100))
        ax.set_xlabel(xplot_label, fontsize=14)
        ax.set_ylabel(yplot_label, fontsize=14)
        #plt.xticks(fontsize=12)
        #plt.yticks(fontsize=12)
        #plt.title(title)
        plt.show()

        return ax


def rotXYZ(R, unit='deg'):
    """ Compute Euler angles from matrix R using XYZ sequence."""

    angles = np.zeros(3)
    angles[0] = np.arctan2(R[2, 1], R[2, 2])
    angles[1] = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    angles[2] = np.arctan2(R[1, 0], R[0, 0])

    if unit[:3].lower() == 'deg':  # convert from rad to degree
        angles = np.rad2deg(angles)

    return angles
