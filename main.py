import torch  #导入了PyTorch库，这是一个流行的深度学习框架
import time   #导入了Python的标准库模块time，用于时间相关操作
from transformers import BertModel, BertTokenizer#从transformers库中导入了BertModel和BertTokenizer。transformers库是Hugging Face提供的用于自然语言处理的库，BERT是一种预训练的语言模型。
import numpy as np#导入了NumPy库，并将其命名为np。NumPy是一个支持大量维度数组与矩阵运算的Python库。
from collections import Counter
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels



base_path = './data'
file_paths = {
    'medical_dev':base_path + '/medical/dev.en',
    'it_dev':base_path + '/it/dev.en',
    'koran_dev':base_path + '/koran/dev.en',
    'subtitles_dev':base_path + '/subtitles/dev.en',
    'law_dev': base_path + '/law/dev.en'
}
model_path = "/mnt/dsw-alitranx-nas/huaike.wc/pretrained_models/transformers/bert-base-uncased/"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'black']






def encode_with_transformers(corpus, model_name_or_path):
    """
    Encodes the corpus using the models in models_to_use. 
    Returns a dictionary from a model name to a list of the encoded sentences and their encodings.
    The encodings are calculatd by average-pooling the last hidden states for each token. 
    """
    print('encoding with BERT')
    states = []

    # Load pretrained model/tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertModel.from_pretrained(model_name_or_path)
    model.to(device)
        
    # Encode text
    start = time.time()
    for sentence in corpus:
        input_ids = tokenizer(sentence, max_length=512, truncation=True,return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        with torch.no_grad():
            output = model(input_ids)
            last_hidden_states = output[0]
                
            # avg pool last hidden layer
            squeezed = last_hidden_states.squeeze(dim=0)
            masked = squeezed[:input_ids.shape[1],:]
            avg_pooled = masked.mean(dim=0)
            states.append(avg_pooled.cpu())
                
    end = time.time()
    print('encoded with BERT in {} seconds'.format(end - start))
    np_tensors = [np.array(tensor) for tensor in states]
    states = np.stack(np_tensors)
    return states

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    uniq = unique_labels(y_true, y_pred)
    classes = np.array(classes)[uniq]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    classes = [c.replace('_dev', '') for c in classes]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # And label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig("./main_confusion.pdf", bbox_inches='tight')
    return ax

def map_clusters_to_classes_by_majority(y_train, y_train_pred):
    """
    Maps clusters to classes by majority to compute the Purity metric.
    """
    cluster_to_class = {}
    for cluster in np.unique(y_train_pred):
        # run on indices where this is the cluster
        original_classes = []
        for i, pred in enumerate(y_train_pred):
            if pred == cluster:
                original_classes.append(y_train[i])
        # take majority         
        cluster_to_class[cluster] = max(set(original_classes), key = original_classes.count)
    return cluster_to_class

def fit_kmeans(embeddings, sentences, class_names, clusters=5, confusion=False):
    all_states = []
    all_sents = []
    all_labels = []
    num_classes = len(class_names)

    for idx, label in enumerate(class_names):
        all_states.append(embeddings[label])
        all_sents += sentences[label]
        all_labels += [idx] * len(sentences[label])
    concat_all_embs = np.concatenate(all_states)
    concat_all_labels = np.array(all_labels)

    # Do not split the data - train=test=all (unsupervised evaluation) 
    train_index = list(range(0, concat_all_embs.shape[0]))
    test_index = list(range(0, concat_all_embs.shape[0]))

    X_train = concat_all_embs[train_index]
    y_train = concat_all_labels[train_index]
    X_test = concat_all_embs[test_index]
    y_test = concat_all_labels[test_index]

    n_classes = len(np.unique(y_train))
    if clusters > 0:
        n_clusters = clusters
    else:
        n_clusters = n_classes

    clustering_model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300, n_init=10
    )
    clustering_model.fit(X_train)

    # predict the cluster ids for train         
    y_train_pred = clustering_model.predict(X_train)
        
    # predict the cluster ids for test
    y_test_pred = clustering_model.predict(X_test)

    # map clusters to classes by majority of true class in cluster         
    clusters_to_classes = map_clusters_to_classes_by_majority(y_train, y_train_pred)

    # plot confusion matrix, error analysis         
    if confusion:
        y_pred_by_majority = np.array([clusters_to_classes[pred] for pred in y_train_pred])
        plot_confusion_matrix(y_train, y_pred_by_majority, class_names)
    
    count=0
    for i, pred in enumerate(y_train_pred):
        if clusters_to_classes[pred] == y_train[i]:
            count += 1
    train_accuracy = float(count)/len(y_train_pred) * 100

    correct_count=0
    for i, pred in enumerate(y_test_pred):
        if clusters_to_classes[pred] == y_test[i]:
            correct_count += 1
    test_accuracy = float(correct_count)/len(y_test_pred) * 100

    return test_accuracy


domain_to_encodings = defaultdict(dict)
domain_to_sentences = defaultdict(dict)
for domain_name in file_paths:
    print('encoding {} with transformers...'.format(domain_name))
    file_path = file_paths[domain_name]
    counter = Counter(open(file_path).readlines())
    lines = list(set(open(file_path).readlines())) # eliminate duplicate sentences
    print('found {} lines'.format(len(lines)))
    res = encode_with_transformers(lines, model_path)
    domain_to_encodings[domain_name] = res
    domain_to_sentences[domain_name] = lines

domains = ['it_dev', 'koran_dev', 'subtitles_dev', 'medical_dev', 'law_dev']
num_clusters = 5

accuracy = fit_kmeans(domain_to_encodings, domain_to_sentences, domains, clusters=num_clusters, confusion=True)