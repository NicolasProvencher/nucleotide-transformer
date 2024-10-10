import os
import nucleotide_transformer
import haiku as hk
import jax
import jax.numpy as jnp
from nucleotide_transformer.pretrained import get_pretrained_model
import numpy as np
import h5py
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run nucleotide transformer decoder.')
parser.add_argument('--path', type=str, required=True, help='Path to the model')
parser.add_argument('--model_name', type=str, required=True, help='Name of the model')
parser.add_argument('--chrm', type=str, required=True, help='Chromosome to filter transcripts')
parser.add_argument('--h5_path', type=str, required=True, help='Path to the HDF5 file')

args = parser.parse_args()

path = args.path
model_name = args.model_name
chrm = args.chrm
h5_path = args.h5_path

# Get pretrained model
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name=model_name,
    model_path=path,
    embeddings_layers_to_save=None,
    attention_maps_to_save=None,  #((1, 4), (7, 16)),
    max_positions=2048,
)

embedding_to_save = range(1, config.num_layers + 1)
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name=model_name,
    model_path=path,
    embeddings_layers_to_save=embedding_to_save,
    attention_maps_to_save=None,  #((1, 4), (7, 16)),
    max_positions=2048,
)

forward_fn = hk.transform(forward_fn)

print(config)

file = h5py.File(h5_path, 'r')
transcripts = {file['transcript']['id'][i].decode('utf-8'): file['transcript']['sequence'][i].decode('utf-8') for i, j in enumerate(file['transcript']['id']) if file['transcript']['chrm'][i].decode('utf-8') == chrm}
file.close()

sequences = [transcripts[k] for k in list(transcripts.keys())]
id = [k for k in list(transcripts.keys())]
print(sequences[0])
print(id[0])
tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

# Initialize random key
random_key = jax.random.PRNGKey(0)

# Infer
outs = forward_fn.apply(parameters, random_key, tokens)

# Assuming `outs` contains the embeddings and `ids` contains the IDs
with h5py.File('/home/noxatras/Desktop/nutr_tutorial/transcripts_database_test3.h5', 'r+') as hdf:
    for i in embedding_to_save:
        logits = outs[f'embeddings_{i}']  # Adjusted to use the correct index
        group_name = f'embeddings_{i}'
        
        # Check if the group exists
        if group_name in hdf:
            group = hdf[group_name]
        else:
            group = hdf.create_group(group_name)
        
        # Append to or create 'id' dataset
        if 'id' in group:
            id_dataset = group['id']
            id_dataset.resize((id_dataset.shape[0] + len(id),))
            id_dataset[-len(id):] = id
        else:
            group.create_dataset('id', data=id, maxshape=(None,))
        
        # Append to or create 'logits' dataset
        if 'logits' in group:
            logits_dataset = group['logits']
            logits_dataset.resize((logits_dataset.shape[0] + logits.shape[0], logits.shape[1], logits.shape[2]))
            logits_dataset[-logits.shape[0]:] = logits
        else:
            group.create_dataset('logits', data=logits, maxshape=(None, logits.shape[1], logits.shape[2]))
        
        # Append to or create 'chrm' dataset
        if 'chrm' in group:
            chrm_dataset = group['chrm']
            chrm_dataset.resize((chrm_dataset.shape[0] + len(id),))
            chrm_dataset[-len(id):] = [chrm for _ in range(len(id))]
        else:
            group.create_dataset('chrm', data=[chrm for _ in range(len(id))], maxshape=(None,))

print("Embeddings saved successfully.")