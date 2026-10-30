
import json, time, re, unicodedata, sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
import matplotlib.pyplot as plt

SEED=42
OUT=Path('results'); OUT.mkdir(exist_ok=True)
REPOS={'safeguard':'xTRam1/safe-guard-prompt-injection','deepset':'deepset/prompt-injections','gandalf':'Lakera/gandalf_ignore_instructions'}

def norm(s):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(s)).strip().casefold())

def load():
    frames=[]; audit=[]
    for name, repo in REPOS.items():
        ds=load_dataset(repo)
        for part, dataset in ds.items():
            raw=dataset.to_pandas()
            labels=pd.Series(np.ones(len(raw),dtype=int) if name=='gandalf' else raw['label'].astype(int))
            text=raw['text'].fillna('').astype(str)
            tmp=pd.DataFrame({
                'row_id':[f'{name}:{part}:{i}' for i in range(len(raw))],
                'dataset':name,
                'publisher_split':part,
                'text':text.to_numpy(),
                'label':labels.to_numpy()
            })
            frames.append(tmp)
            audit.append({
                'dataset':name,
                'publisher_split':part,
                'rows':len(raw),
                'columns':len(raw.columns),
                'positive':int((labels==1).sum()),
                'negative':int((labels==0).sum()),
                'missing_text':int(raw['text'].isna().sum()),
                'missing_label':0 if name=='gandalf' else int(raw['label'].isna().sum()),
                'exact_duplicate_text_rows':int(text.duplicated().sum()),
                'timestamp':'unavailable (no timestamp field)',
                'entity_id':'unavailable (no entity identifier)'
            })
    return pd.concat(frames,ignore_index=True),pd.DataFrame(audit)

def main():
    start=time.perf_counter()
    df,audit=load()
    audit.to_csv(OUT/'audit_raw.csv',index=False)
    print('\nRAW AUDIT\n',audit.to_string(index=False))

    df['key']=df.text.map(norm)
    invalid=(df.key=='')
    print('Empty text removed:',int(invalid.sum()))
    df=df.loc[~invalid].copy()

    # Prefer primary training rows, then primary held-out rows,
    # then external datasets.
    # Keep each text in exactly one dataset and one split.
    priority={'safeguard':0,'deepset':2,'gandalf':3}
    df['priority']=df.dataset.map(priority)+np.where(
        (df.dataset=='safeguard') & (df.publisher_split=='test'),1,0
    )
    df=df.sort_values(['priority','row_id'],kind='stable')
    repeated=df.key.duplicated(keep='first')
    print('Normalized exact duplicates removed across ALL datasets:',int(repeated.sum()))
    df=df.loc[~repeated].copy()

    primary=df[df.dataset=='safeguard'].copy()
    pool=primary[primary.publisher_split=='train'].copy()
    pubtest=primary[primary.publisher_split=='test'].copy()

    # Stratified validation split; near-duplicate template
    # grouping remains a limitation to document.
    tr_idx,val_idx=train_test_split(
        pool.index,
        test_size=.20,
        random_state=SEED,
        stratify=pool.label
    )

    df['split']='test'
    df.loc[tr_idx,'split']='train'
    df.loc[val_idx,'split']='validation'

    train_keys=set(df.loc[df.split=='train','key'])
    val_keys=set(df.loc[df.split=='validation','key'])
    test_keys=set(df.loc[df.split=='test','key'])

    leakage={
        'train_vs_validation':len(train_keys&val_keys),
        'train_vs_all_tests':len(train_keys&test_keys)
    }
    print('Leakage:',leakage)
    assert leakage=={'train_vs_validation':0,'train_vs_all_tests':0}

    split=df[['row_id','dataset','split','label']].copy()
    split.to_csv('split.csv',index=False)

    train=df[df.split=='train']
    val=df[df.split=='validation']
    test=df[df.split=='test']

    model=Pipeline([
        ('tfidf',TfidfVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )),
        ('logreg',LogisticRegression(
            max_iter=1000,
            random_state=SEED
        ))
    ])

    t=time.perf_counter()
    model.fit(train.text,train.label)
    training_seconds=time.perf_counter()-t

    vp=model.predict_proba(val.text)[:,1]

    # Lowest threshold with validation FPR <= 5%.
    neg=vp[val.label.to_numpy()==0]
    candidates=np.r_[
        np.unique(vp),
        np.nextafter(float(vp.max()),np.inf)
    ]
    eligible=[
        float(c) for c in candidates
        if np.mean(neg>=c)<=.05
    ]
    threshold=min(eligible)

    val_fpr=float(np.mean(neg>=threshold))
    val_recall=float(np.mean(
        vp[val.label.to_numpy()==1]>=threshold
    ))

    print(
        f'\nTHRESHOLD={threshold:.17g}  '
        f'validation FPR={val_fpr:.4f} '
        f'recall={val_recall:.4f}'
    )

    test=test.copy()
    test['score']=model.predict_proba(test.text)[:,1]
    test['test_set']=np.where(
        test.dataset=='safeguard',
        'safeguard_test',
        np.where(
            test.dataset=='deepset',
            'deepset_all',
            'gandalf_all'
        )
    )

    scores=test[['row_id','test_set','label','score']]
    scores.to_csv('scores.csv',index=False)

    # In-domain confusion matrix.
    sub=test[test.test_set=='safeguard_test']
    cm=confusion_matrix(
        sub.label,
        (sub.score>=threshold).astype(int),
        labels=[0,1]
    )

    fig,ax=plt.subplots(figsize=(5,4))
    ax.imshow(cm,cmap='Blues')

    for (i,j),v in np.ndenumerate(cm):
        ax.text(j,i,str(v),ha='center',va='center',fontsize=15)

    ax.set_xticks([0,1],['Benign','Injection'])
    ax.set_yticks([0,1],['Benign','Injection'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Safe Guard held-out test confusion matrix')

    fig.tight_layout()
    fig.savefig(OUT/'confusion_matrix.png',dpi=180)
    plt.close(fig)

    details={
        'seed':SEED,
        'rows_original':int(audit.rows.sum()),
        'rows_retained':len(df),
        'empty_text_removed':int(invalid.sum()),
        'normalized_duplicate_rows_removed':int(repeated.sum()),
        'leakage':leakage,
        'train_rows':len(train),
        'validation_rows':len(val),
        'test_rows':len(test),
        'threshold':threshold,
        'validation_fpr':val_fpr,
        'validation_recall':val_recall,
        'training_seconds':training_seconds,
        'total_seconds':time.perf_counter()-start,
        'python_version':sys.version,
        'packages':{'pandas':pd.__version__},
        'test_sets':{
            k:{
                'rows':len(g),
                'positive':int(g.label.sum()),
                'negative':int((g.label==0).sum()),
                'recall':float(np.mean(
                    g.loc[g.label==1,'score']>=threshold
                )) if (g.label==1).any() else None
            }
            for k,g in test.groupby('test_set')
        }
    }

    (OUT/'run_summary.json').write_text(
        json.dumps(details,indent=2)
    )
    (OUT/'threshold.txt').write_text(
        f'{threshold:.17g}\n'
    )

    print('\nSUMMARY\n',json.dumps(details,indent=2))
    print(
        '\nNEXT: python check_checkpoint.py '
        '--split split.csv --scores scores.csv --threshold',
        f'{threshold:.17g}'
    )

if __name__=='__main__':
    main()
