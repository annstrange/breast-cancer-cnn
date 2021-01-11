import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


plt.style.use('ggplot') # I also like fivethirtyeight'
plt.rcParams.update({'font.size': 16, 'font.family': 'sans'})

def show_9grid_image(train_ds):
    fig = plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")


def plot_training_results(history, epochs, filename):
    if history is None:
        return

    print ('what does transfer hist look like')
    print (history.history)    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(1, 2, 1)

    ax1.plot(epochs_range, acc, label='Training Accuracy')
    ax1.plot(epochs_range, val_acc, label='Validation Accuracy')
    ax1.legend(loc='lower right')
    ax1.set_title('Training and Validation Accuracy')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs_range, loss, label='Training Loss')
    ax2.plot(epochs_range, val_loss, label='Validation Loss')
    ax2.legend(loc='upper right')
    ax2.set_title('Training and Validation Loss')
    plt.show()
    fig.tight_layout()
    fig.savefig('../imgs/' + filename + '.png', dpi = 200)

def my_roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    
    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []

    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()

def plot_roc(X_test, y_test, model, plot_name):
    '''
    Arguments:
        X_test: numpy array
        y_test: numpy array test features and lables for test/holdout data 
        model: A trained classification model
        plot_name: string title
        filename: string for saving the plot
    returns:
        saved ROC plot
    Note: This won    
    '''

    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)
    #n_splits=5
    #kf = KFold(n_splits=n_splits, shuffle=True)
    #y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    #for i, (train_index, test_index) in enumerate(kf.split(X)):

    # Predict probabilities, not classes
    y_predictions = model.predict(X_test)[:, 1]

    y_p = list(zip(y_test, y_predictions))
    print('y vs y_predictions look like {}'.format(y_p[:15]))

    #y_prob[test_index] = clf.predict_proba(X_test)
    # can use sklearn.metrics roc_curve, auc
    fpr, tpr, thresholds = my_roc_curve(y_test, y_predictions)  
    #mean_tpr += interp(mean_fpr, fpr, tpr)
    #mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    label = 'ROC (area = {:.2f})'.format(roc_auc)
    ax.plot(fpr, tpr, lw=1, label=label)

    #mean_tpr /= n_splits
    #mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)
    #plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    #plt.plot(fpr, tpr, 'k--',label='ROC (area = %0.2f)' % mean_auc, lw=2)

    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    savefile= '../imgs/' + plot_name + '.png'
    fig.tight_layout()
    ax.grid(False)
    plt.savefig(savefile, dpi=200)
    plt.show()


def plot_predictions(x, y, y_hat, title):

    # Getting data
    x_blue = np.random.uniform(size=100)
    y_blue = 1.0*x_blue + np.random.normal(scale=0.2, size=100)

    x_red = np.random.uniform(size=100)
    y_red = 1.0 - 1.0*x_red + np.random.normal(scale=0.2, size=100)

    x_green = x
    y_green = y

    # plotting
    fig, ax = plt.subplots()
    ax.scatter(x_blue, y_blue, color="blue")
    ax.scatter(x_red, y_red, color="red")

    savefile= '../imgs/' + title + '.png'
    fig.tight_layout()
    plt.savefig(savefile, dpi=200)