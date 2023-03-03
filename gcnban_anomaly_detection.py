# GCNBAN: Graph Convolution Network with Bilinear Attention Network 

## GCNBAN: Graph Convolutional Network with Bilinear Attention Network
"""

# Import libraries
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, Lambda, Multiply, concatenate, Dense, Dropout, Flatten, Dense, Concatenate, Dot, Flatten
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN
import matplotlib.pyplot as plt

# Load training and testing dataset
df_train = pd.read_csv('../UNSW_NB15_training-set.csv')
df_test = pd.read_csv('../UNSW_NB15_testing-set.csv')
df = pd.concat([df_train, df_test])
df = df.reset_index(drop=True)



# Data Preprocessing
list_drop = ['id','attack_cat']
df.drop(list_drop,axis=1,inplace=True)
df_numeric = df.select_dtypes(include=[np.number])

DEBUG =0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('max = '+str(df_numeric[feature].max()))
        print('75th = '+str(df_numeric[feature].quantile(0.95)))
        print('median = '+str(df_numeric[feature].median()))
        print(df_numeric[feature].max()>10*df_numeric[feature].median())
        print('----------------------------------------------------')
    if df_numeric[feature].max()>10*df_numeric[feature].median() and df_numeric[feature].max()>10 :
        df[feature] = np.where(df[feature]<df[feature].quantile(0.95), df[feature], df[feature].quantile(0.95))

df_numeric = df.select_dtypes(include=[np.number])
df_before = df_numeric.copy()

DEBUG = 0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = '+str(df_numeric[feature].nunique()))
        print(df_numeric[feature].nunique()>50)
        print('----------------------------------------------------')
    if df_numeric[feature].nunique()>50:
        if df_numeric[feature].min()==0:
            df[feature] = np.log(df[feature]+1)
        else:
            df[feature] = np.log(df[feature])

df_numeric = df.select_dtypes(include=[np.number])
df_cat = df.select_dtypes(exclude=[np.number])

DEBUG = 0
for feature in df_cat.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = '+str(df_cat[feature].nunique()))
        print(df_cat[feature].nunique()>6)
        print(sum(df[feature].isin(df[feature].value_counts().head().index)))
        print('----------------------------------------------------')
    
    if df_cat[feature].nunique()>6:
        df[feature] = np.where(df[feature].isin(df[feature].value_counts().head().index), df[feature], '-')

df_cat = df.select_dtypes(exclude=[np.number])
df['proto'].value_counts().head().index
df['proto'].value_counts().index

# Feature Selection
best_features = SelectKBest(score_func=chi2,k='all')

X = df.iloc[:,4:-2]
y = df.iloc[:,-1]
fit = best_features.fit(X,y)

df_scores=pd.DataFrame(fit.scores_)                                     
df_col=pd.DataFrame(X.columns)

feature_score=pd.concat([df_col,df_scores],axis=1)
feature_score.columns=['feature','score']
feature_score.sort_values(by=['score'],ascending=True,inplace=True)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X.head()
feature_names = list(X.columns)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

for label in list(df_cat['state'].value_counts().index)[::-1][1:]:
    feature_names.insert(0,label)
    
for label in list(df_cat['service'].value_counts().index)[::-1][1:]:
    feature_names.insert(0,label)
    
for label in list(df_cat['proto'].value_counts().index)[::-1][1:]:
    feature_names.insert(0,label)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.175, 
                                                    random_state = 0,
                                                    stratify=y)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Plot the histogram
y_train.hist()
plt.xlabel('Class Label')
plt.ylabel('Number of Samples')
plt.show()

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train, y_train  = smote.fit_resample(X_train, y_train)

# Plot the histogram
y_train.hist()
plt.xlabel('Class Label')
plt.ylabel('Number of Samples')
plt.show()

# Build Model
def GCBAN():
    # Input layer
    input_layer = Input(shape=(X_train.shape[1], 1))

    # Graph Convolutional layer
    conv_layer = Conv1D(filters=16, kernel_size=3, strides=1, padding='same', name='GC1')(input_layer)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = Activation('relu', name='GC2')(conv_layer)

    # Bilinear Attention Network
    bilinear_layer = Conv1D(filters=32, kernel_size=3, strides=1, padding='same', name='BAN1')(conv_layer)
    bilinear_layer = BatchNormalization()(bilinear_layer)
    bilinear_layer = Activation('relu', name='BAN2')(bilinear_layer)
    bilinear_layer = Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(bilinear_layer)
    bilinear_layer = BatchNormalization()(bilinear_layer)
    bilinear_layer = Activation('relu')(bilinear_layer)
    bilinear_layer = Conv1D(filters=1, kernel_size=3, strides=1, padding='same', name='BAN3')(bilinear_layer)
    bilinear_layer = BatchNormalization()(bilinear_layer)
    bilinear_layer = Activation('relu', name='BAN4')(bilinear_layer)

    # Softmax layer
    softmax_layer = Lambda(lambda x: K.softmax(K.sum(x, axis=1)))(bilinear_layer)

    # Multiplication layer
    multiply_layer = Multiply()([conv_layer, softmax_layer])

    # Flatten layer
    flatten_layer = Flatten()(multiply_layer)

    # Dense layer
    dense_layer = Dense(64, activation='relu')(flatten_layer)
    dense_layer = Dropout(0.5)(dense_layer)

    # Output layer
    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    # Define and Compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    op = Adam(learning_rate=1e-04, 
                              epsilon=1e-07)
    model.compile(loss='binary_crossentropy', 
                  optimizer=op, 
                  metrics='accuracy')

    return model

# Train the model
model = GCBAN()
path = '../model-GCNBAN' + str(model.name)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=41) 
save = ModelCheckpoint(path+'.h5', monitor='val_loss', mode='min', save_best_only=True)
callbacks = [save,es]
history = model.fit(X_train, y_train, batch_size=512, epochs=200, validation_data=(X_test, y_test), callbacks=callbacks)

# Visualization
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score, precision_recall_curve
import seaborn as sns

# Predict on test set
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='GnBu', fmt='g', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = {:.2f})'.format(auc_score))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# F1-score, Recall, Precision, Accuracy
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("F1-score: {:.2f}".format(f1))
print("Recall: {:.2f}".format(recall))
print("Precision: {:.2f}".format(precision))

# Calculate AUC-ROC and AUC-PR
y_pred = model.predict(X_test)
auc_roc = roc_auc_score(y_test, y_pred)
print('AUC-ROC:', auc_roc)

auc_pr = average_precision_score(y_test, y_pred)
print('AUC-PR:', auc_pr)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_roc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot PR curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % auc_pr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower left")
plt.show()

history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['accuracy', 'val_accuracy']].plot()
history_df.loc[0:, ['loss', 'val_loss']].plot()
print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_accuracy'].max()))