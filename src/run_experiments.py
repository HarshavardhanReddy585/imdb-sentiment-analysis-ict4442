"""CPU-friendly, reproducible Phase 2 baseline experiments.

Implements two approved architecture families without requiring a deep-learning
framework: an embedding-average MLP and a 1D convolutional n-gram model.
This is a transparent interim baseline; final submission should add BiLSTM and
DistilBERT and rerun the common evaluation protocol.
"""
from __future__ import annotations
import argparse, csv, json, random, re, tarfile
from pathlib import Path
import numpy as np

TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
MAX_LEN = 40

def read_reviews(root: Path, limit: int | None = None):
    rows = []
    for label, name in [(0, 'neg'), (1, 'pos')]:
        files = sorted((root / name).glob('*.txt'))
        if limit:
            files = files[: limit // 2]
        for f in files:
            rows.append((f.read_text(encoding='utf-8', errors='ignore'), label))
    return rows

def tok(text):
    return TOKEN.findall(text.lower())

def extract_archive(archive: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'aclImdb').exists(): return target / 'aclImdb'
    with tarfile.open(archive, 'r:gz') as t: t.extractall(target)
    return target / 'aclImdb'

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    return {'accuracy': float((p == y).mean()), 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / max(precision + recall, 1e-12),
            'confusion_matrix': [[tn, fp], [fn, tp]]}

def batches(n, size, rng):
    idx = np.arange(n); rng.shuffle(idx)
    for i in range(0, n, size): yield idx[i:i+size]

def vectorize(rows, vocab, max_len=MAX_LEN):
    X = np.zeros((len(rows), max_len), dtype=np.int32); y=[]
    for i, (text, label) in enumerate(rows):
        ids = [vocab.get(w, 1) for w in tok(text)[:max_len]]
        X[i, :len(ids)] = ids; y.append(label)
    return X, np.asarray(y, dtype=np.float32)

def train_mlp(Xtr, ytr, Xva, yva, V, seed=42, epochs=10):
    rng=np.random.default_rng(seed); d=48
    E=rng.normal(0,.08,(V,d)); W1=rng.normal(0,.05,(d,32)); b1=np.zeros(32); W2=rng.normal(0,.05,32); b2=0.
    lr=.08
    def forward(X):
        mask=(X>1)[...,None]; h=(E[X]*mask).sum(1)/np.maximum(mask.sum(1),1); z=h@W1+b1; a=np.tanh(z); q=sigmoid(a@W2+b2); return h,z,a,q
    for _ in range(epochs):
        for ix in batches(len(Xtr), 64, rng):
            h,z,a,q=forward(Xtr[ix]); n=len(ix); dy=(q-ytr[ix])/n
            dW2=a.T@dy; db2=dy.sum(); da=dy[:,None]*W2; dz=da*(1-a*a); dW1=h.T@dz; db1=dz.sum(0); dh=dz@W1.T
            dE=np.zeros_like(E)
            for j,row in enumerate(Xtr[ix]):
                ids=row[row>1];
                if len(ids): np.add.at(dE,ids,dh[j]/len(ids))
            W2-=lr*dW2; b2-=lr*db2; W1-=lr*dW1; b1-=lr*db1; E-=lr*dE
    _,_,_,q=forward(Xva); return metrics(yva,(q>=.5).astype(int))

def train_cnn(Xtr, ytr, Xva, yva, V, seed=42, epochs=10):
    rng=np.random.default_rng(seed); d=16; k=3; F=4
    E=rng.normal(0,.08,(V,d)); K=rng.normal(0,.08,(F,k,d)); W=rng.normal(0,.05,F); b=0.; lr=.08
    def features(X):
        emb=E[X]; out=np.zeros((len(X),F))
        for i in range(len(X)):
            valid=(X[i]>1).astype(float)
            for f in range(F):
                vals=[]
                for j in range(max(1, len(X[i])-k+1)):
                    if valid[j:j+k].sum() == k: vals.append((emb[i,j:j+k]*K[f]).sum())
                out[i,f]=max(vals) if vals else 0.
        return out
    # Train classifier over fixed convolutional features for a stable CPU interim run.
    A=features(Xtr); B=features(Xva)
    for _ in range(epochs):
        for ix in batches(len(A),64,rng):
            q=sigmoid(A[ix]@W+b); dy=(q-ytr[ix])/len(ix); W-=lr*A[ix].T@dy; b-=lr*dy.sum()
    return metrics(yva,(sigmoid(B@W+b)>=.5).astype(int))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--archive', required=True); ap.add_argument('--max-train',type=int,default=4000); ap.add_argument('--max-test',type=int,default=1000); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); project=Path(__file__).parents[1]; data=extract_archive(Path(a.archive),project/'data'); out=project/'results'; out.mkdir(exist_ok=True)
    alltr=read_reviews(data/'train',a.max_train); test=read_reviews(data/'test',a.max_test); random.shuffle(alltr); cut=int(.8*len(alltr)); tr,va=alltr[:cut],alltr[cut:]
    counts={};
    for text,_ in tr:
        for w in set(tok(text)): counts[w]=counts.get(w,0)+1
    words=sorted(counts,key=counts.get,reverse=True)[:12000]; vocab={'<pad>':0,'<unk>':1}; vocab.update({w:i+2 for i,w in enumerate(words)})
    Xtr,ytr=vectorize(tr,vocab); Xva,yva=vectorize(va,vocab); Xte,yte=vectorize(test,vocab)
    results={'dataset':{'train':len(tr),'validation':len(va),'test':len(test),'vocab':len(vocab),'max_length':MAX_LEN,'seed':a.seed},'models':{}}
    results['models']['MLP_embedding_average']={'validation':train_mlp(Xtr,ytr,Xva,yva,len(vocab),a.seed)}
    results['models']['CNN_1D_maxpool']={'validation':train_cnn(Xtr,ytr,Xva,yva,len(vocab),a.seed)}
    (out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['model','split','accuracy','precision','recall','f1'])
        for model,v in results['models'].items():
            m=v['validation']; w.writerow([model,'validation',m['accuracy'],m['precision'],m['recall'],m['f1']])
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
