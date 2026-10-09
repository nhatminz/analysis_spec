"""FastGRPO ShareGPT collation (assistant mask includes role delimiters)."""
import torch

class DataCollator:
    def __init__(self, tokenizer, max_length=4096, model_type='qwen2'):
        self.tokenizer, self.max_length, self.model_type = tokenizer, max_length, model_type

    def __call__(self, batch):
        rows, masks = [], []
        for example in batch:
            if self.model_type in ('qwen2', 'qwen3'):
                system='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
                def render(role, content): return '<|im_start|>'+role+'\n'+content+'<|im_end|>\n'
            elif self.model_type == 'llama':
                system='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>'
                def render(role, content): return '<|start_header_id|>'+role+'<|end_header_id|>\n\n'+content+'<|eot_id|>'
            else:
                raise ValueError('supported model types: qwen2, qwen3, llama')
            ids=self.tokenizer.encode(system,add_special_tokens=False); mask=[0]*len(ids)
            for conversation in example['conversations']:
                role={'human':'user','gpt':'assistant'}.get(conversation['from'],conversation['from'])
                tokens=self.tokenizer.encode(render(role,conversation['value']),add_special_tokens=False)
                ids.extend(tokens); mask.extend([int(role in ('assistant','ASSISTANT'))]*len(tokens))
            rows.append(ids[:self.max_length]); masks.append(mask[:self.max_length])
        length=max(map(len,rows))
        return dict(input_ids=torch.tensor([x+[self.tokenizer.eos_token_id]*(length-len(x)) for x in rows]),
                    attention_mask=torch.tensor([[1]*len(x)+[0]*(length-len(x)) for x in rows]),
                    loss_mask=torch.tensor([x+[0]*(length-len(x)) for x in masks]))
