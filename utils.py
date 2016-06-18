import simplejson as json
import numpy as np
import math
import uuid

def score(y,o):
    tp = fp = fn = tn = 0
    for i,j in zip(y,o):
        i = int(i[0])
        j = float(j[0])
        if i == 1 and j > .5:
            tp += 1
        elif i == 1 and j <= .5:
            fp += 1
        elif i == 0 and j <= .5:
            tn += 1
        elif i == 0 and j > .5:
            fn += 1
    pre = tp/(tp+fp)
    re = tp/(tp+fn)
    se = tp/(tp+fn)
    accuracy = 2*tp/(2*tp + fp + fn)
    f1 = 2* ((pre*re)/(pre+re))
    toprint = (round(pre,3), round(re,3), round(se,3), round(accuracy,3), round(f1,3), tp, tn, fp, fn)
    print(toprint)
    return toprint


def preprocess(x,y):
    a = np.genfromtxt(x, delimiter=",")
    b = np.genfromtxt(y, delimiter=",")
    start = int(a[1][0])
    end = int(a[-1][0])
    results = []
    for i,j in zip(range(start, end), range(1,len(a))):
        results.append((b[i, [x for x in range(4,27) if x!= 19]], sum(a[j, [x for x in range (8,19)]])))
    np.random.shuffle(results)
    x_input = [x[0] for x in results]
    y_input = [[x[1]] for x in results]
    return x_input, y_input


def save_score(filename, scores):
    with open('outputs/'+filename, "w") as f:
        f.write("pass, precision, recall, sensitivity, accuracy, f1_score, true positive, true negative, false positive, false negative\n")
        for i, args in enumerate(scores, 1):
            f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n'.format(i,*args))


def sigmoid(x):
        return 1/(1+np.exp(-x))


def dsigmoid(x):
        return x*(1-x)


def simple_id():
    with open('config.json', 'r') as f:
        data = json.load(f)
        i = int(data["iteration"])
        i += 1
        new_data = {"iteration": i} 
    with open('config.json', 'w') as f:
        json.dump(new_data, f)
    return str(i)


def id_gen():
    a = uuid.uuid4().int
    while (len(str(a)) > 6):
        z = int(len(str(a))/2)-1
        b = int(str(a)[0:z])
        c = int(str(a)[z:-1])
        a = b^c
    return(np.base_repr(a,36).lower())

