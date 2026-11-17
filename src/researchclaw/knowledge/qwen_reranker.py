"""CUDA-only Qwen3 official yes/no scoring, with explicit token windows."""
import time
from .token_safety import source_windows

PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
          'based on the Query and the Instruct provided. Note that the answer '
          'can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


class QwenReranker:
    def __init__(self, *, revision, model_path=None, batch_size=4, max_length=2048, optimized=True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; Qwen CPU fallback disabled')
        self.model_name = 'Qwen/Qwen3-Reranker-0.6B'
        self.model_revision = revision
        self.device = 'cuda:0'
        self.batch_size, self.max_length = batch_size, max_length
        if batch_size < 1 or max_length < 128:
            raise ValueError('Positive batch size and max_length >=128 required')
        self.optimized = optimized
        path = model_path or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, padding_side='left', local_files_only=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(path, revision=revision, torch_dtype=dtype,
                                                        attn_implementation='sdpa', local_files_only=True).to(self.device).eval()
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.no, self.yes = [self.tokenizer.convert_tokens_to_ids(word) for word in ('no', 'yes')]
        self.last_diagnostics = {}

    def score(self, query, documents):
        return self.score_windowed(query, documents)

    def score_windowed(self, query, documents):
        import torch
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        prompt = f'<Instruct>: {instruction}\n<Query>: {query}\n<Document>: '
        budget = self.max_length - len(self.prefix) - len(self.suffix)
        windows, owners = [], []
        for i, document in enumerate(documents):
            spans = source_windows(document or ' ', self.tokenizer, prefix=prompt, maximum=budget)
            for span in spans:
                ids = self.tokenizer.encode(prompt + span['text'], add_special_tokens=False)
                windows.append(self.prefix + ids + self.suffix)
                owners.append(i)
                if len(windows)>getattr(self,'maximum_windows',float('inf')):
                    raise ValueError('Reranker request exceeds configured window limit')
        scores = [float('-inf')] * len(documents)
        if self.optimized:
            order = sorted(range(len(windows)),key=lambda i:len(windows[i]))
            windows = [windows[i] for i in order]
            owners = [owners[i] for i in order]
        started = time.perf_counter()
        with torch.inference_mode():
            for offset in range(0,len(windows),self.batch_size):
                inputs = self.tokenizer.pad({'input_ids': windows[offset:offset+self.batch_size]},
                                            padding=True, return_tensors='pt').to(self.device)
                kwargs = {'logits_to_keep':1} if self.optimized else {}
                logits = self.model(**inputs, use_cache=False, **kwargs).logits[:, -1, :]
                # logit difference is monotonic with official softmax([no,yes]).
                values = (logits[:, self.yes].float() - logits[:, self.no].float()).tolist()
                for owner, value in zip(owners[offset:offset+self.batch_size], values):
                    scores[owner] = max(scores[owner], value)
        torch.cuda.synchronize()
        self.last_diagnostics = {'device': self.device, 'windows': len(windows),
                                 'seconds': time.perf_counter()-started, 'max_length':self.max_length,
                                 'optimized':self.optimized,'batch_size':self.batch_size}
        return scores
