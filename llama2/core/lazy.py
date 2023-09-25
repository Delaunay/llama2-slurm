
import os
import json
from pathlib import Path
from functools import cache
from contextlib import ExitStack

import torch

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer


import torch.serialization as torchsave


load_module_mapping = {
    'torch.tensor': 'torch._tensor'
}

import pickle


@cache
def make_storage_type(name):
    return torchsave.StorageType(name)


class LazyTorchStateLoader:
    def __init__(self, filename, map_location) -> None:
        self.loaded_storages = {}
        self.restore_location = torchsave._get_restore_location(map_location)
        self.override_class = {
            ('torch._utils', '_rebuild_tensor_v2'): self.lazy_rebuild
        }

        def lazy_persistent_load(saved_id):
            typeame, storage_type, key, location, numel = saved_id
            return saved_id
        
        self.lazy_persistent_load = lazy_persistent_load
       
        class UnpicklerWrapper(pickle.Unpickler):
            override_class = None
            
            def find_class(self, mod_name, name):
                overriden = self.override_class.get((mod_name, name))
                if overriden is not None:
                    original = super().find_class(mod_name, name)
                    return overriden(original)
                
                if type(name) is str and 'Storage' in name:
                    try:
                        return make_storage_type(name)
                    except KeyError:
                        pass
                mod_name = load_module_mapping.get(mod_name, mod_name)
                return super().find_class(mod_name, name)
            
        self.unpickler_type = UnpicklerWrapper
        self.stack = ExitStack()
        
        self.filename = filename
        self.file = None
        self.zipfile = None
        
    def __enter__(self):
        self.file = self.stack.enter_context(torchsave._open_file_like(self.filename, 'rb'))
        
        if not torchsave._is_zipfile(self.file):
            raise RuntimeError("Not zip")
            
        self.zipfile = self.stack.enter_context(torchsave._open_zipfile_reader(self.file))
            
        if torchsave._is_torchscript_zip(self.zipfile):
            raise RuntimeError("Wrong load function")
        
        return self
        
    def __exit__(self, *args):
        self.stack.close()
        
    def unpickle(self, data_file='data.pkl'):
        import io
        
        data = io.BytesIO(self.zipfile.get_record(data_file))
        
        unpickler = self.unpickler_type(data)
        unpickler.override_class = self.override_class
        unpickler.persistent_load = self.lazy_persistent_load
        return unpickler.load()

    def lazy_rebuild(self, original):
        def _wrapper(*args, **kwargs):
            # original: _rebuild_tensor_v2
            # args: (
            #   ('storage', <torch.serialization.StorageType object at 0x7f9736729a60>, '1', 'cpu', 158607360) <= saved_id should a storage
            #
            #   13107200,          storage_offset
            #
            #   (2560, 5120),       size
            #   (5120, 1),          stride
            #   False,              require_grad
            #   OrderedDict()       backward_hooks ?
            #)
            def load_tensor():
                storage = self.persistent_load(args[0])
                return original(storage, *args[1:], **kwargs)

            return load_tensor
            
        return _wrapper

    def load_tensor(self, dtype, numel, key, location):
        name = f'data/{key}'

        storage = self.zipfile.get_storage_from_record(name, numel, torch.UntypedStorage)._typed_storage()._untyped_storage
        # TODO: Once we decide to break serialization FC, we can
        # stop wrapping with TypedStorage
        typed_storage = torch.storage.TypedStorage(
            wrap_storage=self.restore_location(storage, location),
            dtype=dtype,
            _internal=True)

        return typed_storage

    def persistent_load(self, saved_id):
        assert isinstance(saved_id, tuple)
        typename = torchsave._maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        assert typename == 'storage', \
            f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, numel = data
        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype

        if key in self.loaded_storages:
            typed_storage = self.loaded_storages[key]
        else:
            nbytes = numel * torch._utils._element_size(dtype)
            typed_storage = self.load_tensor(dtype, nbytes, key, torchsave._maybe_decode_ascii(location))

        return typed_storage


class LazyModelParallelLoader:
    def __init__(self, folder, map_location="cpu") -> None:
        self.checkpoints = sorted(Path(folder).glob("*.pth"))
        self.stack = ExitStack()
        self.map_location = map_location
        self.weigths = dict()
        self.loader = dict()

    def __enter__(self):
        for name in self.checkpoints:
            _, id, ext = str(name).split('.')
            
            lazy_loader = LazyTorchStateLoader(name, self.map_location)
            self.stack.enter_context(lazy_loader)
            
            lazy_weights = lazy_loader.unpickle()
            self.weigths[int(id)] = lazy_weights
        
        return self
    
    def no_cache_weight(self, id):
        return self.loader[id].unpicke()
    
    def get_weight(self, key, device):
        return self.weigths[device][key]
    
    def split_count(self):
        return len(self.weigths)
    
    def stiched_weight(self, dim):
        pass

    def keys(self):
        """Keep the weight"""
        keys = dict()
        
        for _, w in self.weigths.items():
            for k in w.keys():
                keys[k] = 1
        
        return list(keys.keys())

    def __exit__(self, *args):
        self.stack.close()
        return

    
def lazy(method, *args, **kwargs):
    return (method, args, kwargs)
    
def lazy_model_maker(params: ModelArgs):
    import torch.nn.functional as F
    from fairscale.nn.model_parallel.layers import (
        ColumnParallelLinear,
        ParallelEmbedding,
        RowParallelLinear,
    )
    from torch import nn
    from llama.model import TransformerBlock, RMSNorm
    
    modules = []
    
    vocab_size = params.vocab_size
    n_layers = params.n_layers
    
    modules.append(lazy(
        ParallelEmbedding,              #
        params.vocab_size,              #
        params.dim,                     #
        init_method=lambda x: x         #
    ))
    
    for layer_id in range(params.n_layers):
        modules.append(lazy(TransformerBlock, layer_id, params))
        
    modules.append(lazy(RMSNorm, params.dim, eps=params.norm_eps))
    
    modules.append(lazy(
        ColumnParallelLinear,
        params.dim, 
        params.vocab_size, 
        bias=False, 
        init_method=lambda x: x
    ))

    
def read_checkpoint(ckpt_dir, max_seq_len, max_batch_size):
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    
    with LazyModelParallelLoader(ckpt_dir, "cpu") as lazy:
        keys = lazy.keys()
        
        for k in keys:
            data = []
            
            for i in range(lazy.split_count()):
                lzyw = lazy.get_weight(k, i)
                
                data.append(lzyw())
        
            print(f"{k:>40} {(data[0] - data[1]).abs().sum().item():10.2f} {data[0].shape}")
    

def get_checkpoints(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    """LLAMA makes one checkpoint per GPU
    
    #  7b needs 1 GPUs
    # 13b needs 2 GPUs
    # 70b needs 8 GPUs
    """

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())


    tokenizer = Tokenizer(model_path=tokenizer_path)
    
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        **params,
    )
    model_args.vocab_size = tokenizer.n_words
    
    model = Transformer(model_args)
    
    # ckpt_path = checkpoints[get_model_parallel_rank()]
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # model.load_state_dict(checkpoint, strict=False)
    
    # Llama(model, tokenizer)


def main():
    ROOT = os.path.dirname(__file__)

    MODEL_NAME = 'llama-2-13b'

    ckpt_dir: str = os.path.join(ROOT, 'llama', MODEL_NAME)
    tokenizer_path: str = os.path.join(ROOT, 'llama', 'tokenizer.model')


    temperature: float = 0.6
    top_p: float = 0.9
    max_seq_len: int = 512
    max_batch_size: int = 1
    max_gen_len: int = None

    read_checkpoint(ckpt_dir)



if __name__ == '__main__':
    main()
