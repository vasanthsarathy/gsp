import json
from pathlib import Path
import random
import click

def load_original_data(ctx):
    with open(ctx['input'], "r") as f:
        data = json.load(f)

    click.secho(f"Total data: {len(data['utterances'])}")
    if ctx['num_data'] == -1:
        for obj in data['utterances']:
            yield obj
    else:   
        random_data = random.sample(data['utterances'], ctx['num_data'])
        for obj in random_data:
            yield obj

def load_styled_data(ctx):
    
    data = load_jsonl(ctx['input'])
    if ctx['num_data'] == -1:
        for obj in data:
            yield obj
    else:
        random_data = random.sample(data, ctx['num_data'])
        for obj in random_data:
            yield obj



def record_objs(stream, name, ctx):
    """
    Wraps a stream with logic to record each object to a JSON file and yield it
    again for further processing.
    """
    #print(f"// Saving rows to file")
    file_stem = Path(ctx['input']).stem

    output_filepath = f"gsp/results/{file_stem}_{name}.jsonl"

    with open(output_filepath, 'wt') as out_file:
        for obj in stream:
            json.dump(obj, out_file, default=str)
            out_file.write('\n')
            yield obj


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


