#--------------- ROC Plot and AUC---------------
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

#--------------PolyU Dataset-------------

X, y = make_classification(n_samples=3744, n_classes=2, random_state=2) # generate 2 class dataset
trainX, testX, trainy1, testy1 = train_test_split(X, y, test_size=0.2, random_state=2) # split into train/test sets
rp_probs = [0 for _ in range(len(testy1))] # generate a random prediction (majority class)
model = LogisticRegression(solver='lbfgs') # fit a model
model.fit(trainX, trainy1) # fit a model
lr_probs = model.predict_proba(testX) # predict probabilities
lr_probs = lr_probs[:, 1] # keep probabilities for the positive outcome only
rp_auc = roc_auc_score(testy1, rp_probs)
lr_auc = roc_auc_score(testy1, lr_probs)
print('PolyU study: ROC AUC=%.3f' % (lr_auc))
#rp_fpr, rp_tpr, _ = roc_curve(testy1, rp_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy1, lr_probs)
pyplot.plot(lr_fpr, lr_tpr, 'm', label='PolyU (AUC = %.3f)' %lr_auc)

#-----------------CASIA Dataset------------------------------------

X, y = make_classification(n_samples=7200, n_classes=2, random_state=2)
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2)
rp_probs = [0 for _ in range(len(testy))]
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
lr_probs = model.predict_proba(testX)
lr_probs = lr_probs[:, 1]
# calculate scores
#rp_auc = roc_auc_score(testy, rp_probs)
lr_auc = roc_auc_score(testy, lr_probs)
print('CASIA study: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
rp_fpr, rp_tpr, _ = roc_curve(testy, rp_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
pyplot.plot(lr_fpr, lr_tpr,'y', label=' CASIA (AUC = %.3f)' %lr_auc)

#--------------FV-SUM Dataset-------------

X, y = make_classification(n_samples=5940, n_classes=2, random_state=2)
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2)
rp_probs = [0 for _ in range(len(testy))]
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
lr_probs = model.predict_proba(testX)
lr_probs = lr_probs[:, 1]
# calculate scores
#rp_auc = roc_auc_score(testy, rp_probs)
lr_auc = roc_auc_score(testy, lr_probs)
print('FV-SUM study: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
rp_fpr, rp_tpr, _ = roc_curve(testy, rp_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
pyplot.plot(lr_fpr, lr_tpr,'c', label=' FV-SUM (AUC = %0.3f)' %lr_auc)

#--------------VERA Dataset-------------

X, y = make_classification(n_samples=440, n_classes=2, random_state=2)
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=2)
rp_probs = [0 for _ in range(len(testy))]
model = LogisticRegression(solver='lbfgs')
model.fit(trainX, trainy)
lr_probs = model.predict_proba(testX)
lr_probs = lr_probs[:, 1]
# calculate scores
#rp_auc = roc_auc_score(testy, rp_probs)
lr_auc = roc_auc_score(testy, lr_probs)
print('VERA study: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
rp_fpr, rp_tpr, _ = roc_curve(testy, rp_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
pyplot.plot(lr_fpr, lr_tpr,'k', label=' VERA (AUC = %0.3f)' %lr_auc)
#---------------------------------------
# axis labels
pyplot.title("ROC Curve")
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend() # show the legend
pyplot.savefig('ROC study for various dataset.jpg', dpi=1200, bbox_inches='tight') # save the plot as figure
pyplot.show() # show the plot
#-----------------------------------------------------
