from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from motivation.data import extract_row,prepare,normalized_question,tokenize
from motivation.config import Config
from motivation.compatibility import validate_model_config,QWEN_3B

class Tokenizer:
    chat_template='<|im_start|>{{ role }}{{ content }}'
    padding_side='left';pad_token_id=151643;eos_token_id=151645
    def apply_chat_template(self,messages,**kw):
        assert kw==dict(tokenize=False,add_generation_prompt=True)
        return ''.join(m['role']+':'+m['content'] for m in messages)+'assistant:'
    def encode(self,text,**kw):return list(text.encode())


def files(tmp_path,train_n=8,test_n=70):
    def rows(n,split):
        return [{'prompt':np.array([dict(role='user',content=f"Question:\n{split} question {i}\nAnswer:\nLet's think step by step.\n")]),
                 'reward_model':dict(ground_truth=str(i)), 'unique_id':f'{split}-{i}'} for i in range(n)]
    paths=[tmp_path/'train.parquet',tmp_path/'test.parquet']
    for path,n,split in zip(paths,[train_n,test_n],['train','test']):pd.DataFrame(rows(n,split)).to_parquet(path)
    return paths


def test_real_schema_extraction_prompt_precedes_raw_question():
    row=extract_row(dict(question='different raw field',prompt=np.array([dict(role='user',content='Question:\n2 + 2?\nAnswer:\nLet\'s think step by step.\n')]),reward_model={'ground_truth':'4'}),'train',0)
    assert normalized_question(row['question'])=='2 + 2?';assert row['answer']=='\\boxed{4}'
    assert row['id']=='train:0'


def test_fixed_64_first16_and_no_rng_mutation(tmp_path):
    import random
    from motivation.state import digest
    train,test=files(tmp_path);before=random.getstate()
    a=prepare(train,test,tmp_path/'a',Tokenizer());b=prepare(train,test,tmp_path/'a',Tokenizer())
    assert digest(before)==digest(random.getstate());assert a[2]==b[2]
    assert len(a[1])==64;assert len({r['id'] for r in a[1]})==64
    assert [x['ordinary_measurement'] for x in a[2]['subset']]==[True]*16+[False]*48
    assert all(r['split']=='test' for r in a[1]);assert all(r['split']=='train' for r in a[0])
    assert a[2]['eval_responses_per_prompt']==1


def test_overlap_fail_and_within_duplicates_recorded(tmp_path):
    train,test=files(tmp_path)
    d=pd.read_parquet(train);d=pd.concat([d,d.iloc[:1]],ignore_index=True);d.to_parquet(train)
    _,_,m=prepare(train,test,tmp_path/'ok',Tokenizer());assert len(m['train']['duplicates'])==1
    items=pd.read_parquet(test).to_dict('records');items[0]['prompt']=list(d.iloc[0]['prompt'])
    pd.DataFrame(items).to_parquet(test)
    with pytest.raises(ValueError,match='leakage'):prepare(train,test,tmp_path/'bad',Tokenizer())


def test_not_enough_unique_and_missing_paths(tmp_path):
    train,test=files(tmp_path,test_n=63)
    with pytest.raises(ValueError,match='at least 64'):prepare(train,test,tmp_path/'o',Tokenizer())
    with pytest.raises(FileNotFoundError):prepare(tmp_path/'absent.parquet',test,tmp_path/'o',Tokenizer())


def test_subset_manifest_cannot_change(tmp_path):
    train,test=files(tmp_path);prepare(train,test,tmp_path/'o',Tokenizer())
    with pytest.raises(ValueError,match='Stored evaluation'):prepare(train,test,tmp_path/'o',Tokenizer(),42)


def test_3b_family_validation_and_accumulation_fail_closed():
    good=dict(QWEN_3B,rope_theta=1000000.)
    validate_model_config(good)
    for hidden in (1536,3584,5120):
        with pytest.raises(ValueError,match='3B'):validate_model_config(dict(good,hidden_size=hidden))
    with pytest.raises(ValueError,match='accumulation'):Config(accumulation_steps=4).validate()
    with pytest.raises(ValueError,match='completed target'):Config(train_steps=2).validate()
    assert Config().n_eval(20)==16 and Config().n_eval(100)==64


def test_non_simplelr_and_swapped_official_splits_rejected():
    row=dict(prompt='q',reward_model={'ground_truth':'1'},data_source='dapo')
    with pytest.raises(ValueError,match='not SimpleLR'):extract_row(row,'train',0)
    row['data_source']='simplelr_abel';row['extra_info']={'split':'test'}
    with pytest.raises(ValueError,match='split swap'):extract_row(row,'train',0)
