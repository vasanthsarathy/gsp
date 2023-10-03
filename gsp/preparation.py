# Module for preparing the data for finetuning.

from gsp.config.prompts import *
from pathlib import Path
import gsp.io as io
import pandas as pd 
import json
import click

def prepare(ctx):
    """
    This pipeline converts a json dataset to several format suitable for finetuning an LLM 
    """
    data = io.load_jsonl(ctx['input'])
    
    items = []
    for obj in data:
        if "augmented_utterance" in obj:
            # use augmented utterance if available
            if not obj["augmented_utterance"] == "":
                item = {'utterance': obj['augmented_utterance'],'goal_semantics': obj['semantics']}
        elif "styled_utterance" in obj:
            # see the stylize() function for the key-values in the stylized jsons 
            item = {'utterance': obj['styled_utterance'],'goal_semantics': obj['semantics']}
        elif "base_utterance" in obj:
            item = {'utterance': obj['base_utterance'],'goal_semantics': obj['semantics']}
        else:
            # if the data has not been stylized then use old keyvalues
            item = {'utterance': obj['utteranceText'], 'goal_semantics': obj['desiredSemantics']}

        item['json_semantics'] = deconstruct(item['goal_semantics'])
        item['intent'] = item['json_semantics']['intent']
        item['central_proposition'] = item['json_semantics']['central_proposition']
        item['supplemental_semantics'] = item['json_semantics']['supplemental_semantics']
        
        # Stylizer info 
        if 'stylizer' in obj:
            item['stylizer'] = obj['stylizer']
        else:
            item['stylizer'] = ""

        # augmentation_info
        if 'augmentation_info' in obj:
            item['augmenter'] = obj['augmentation_info']['augmenter']
            if 'language' in obj['augmentation_info']:
                item['language'] = obj['augmentation_info']['language']
            else:
                item['language'] = "en"
        else:
            item['augmenter'] = ""


        # let's get the robot repertoire 

        if 'robot_repertoire' in obj:
            actions = []
            for action in obj['robot_repertoire']['actions']:
                actions.append(action['name'])
            item['actions'] = actions

            properties = []
            for property in obj['robot_repertoire']['properties']:
                properties.append(property['name'])
            item['properties'] = properties
        elif "promptInfo" in obj:
            actions = []
            for action in obj['promptInfo']['actions']:
                actions.append(action['name'])
            item['actions'] = actions

            properties = []
            for property in obj['promptInfo']['properties']:
                properties.append(property['name'])
            item['properties'] = properties

        if ctx['context']:
            # if the context flag is set. 
            item['prompt_template'] = prompt_template_deconstructed_goal_semantics_with_context
            item['text'] = item['prompt_template'].format(instruction_with_context=instruction_with_context,
                                                          example_with_context=example_with_context,
                                                          utterance=item['utterance'],
                                                          actions=item['actions'],
                                                          properties=item['properties'],
                                                          output=item['json_semantics'])

            item['input'] = item['utterance'] + "\n\nAvailable actions:\n" + str(item['actions']) + "\n\nAvailable detectors:\n" + str(item['properties']) 
            item['output'] = item['json_semantics']
            item['instruction'] = instruction_with_context + "\n\n" + example_with_context.format(actions=item['actions'],
                                                                                                  properties=item['properties'])

        else:
            item['prompt_template'] = prompt_template_deconstructed_goal_semantics
            item['text'] = item['prompt_template'].format(instruction=instruction,
                                                        example=example,
                                                        input=item['utterance'], 
                                                        output=item['json_semantics'])
            item['input'] = item['utterance']
            item['output'] = item['json_semantics']
            item['instruction'] = instruction+"\n\n"+example

        items.append(item)
        
    file_stem = Path(ctx['input']).stem
    output_filepath = f"gsp/results/{file_stem}_finetuning_context-{ctx['context']}_{ctx['purpose']}.csv"

    df = pd.DataFrame(items)
    df_pruned = df.drop_duplicates(subset=['utterance'], keep='first')
    click.secho("Some example entries:")
    print(df.head())
    df_pruned.to_csv(output_filepath, index=False)
    click.secho(f"Dataset (with {len(items)} samples) saved as {output_filepath} ready for use in finetuning your LLM!", fg="green")
    return True

def deconstruct(text):
    first = text.split("(")
    intent = first[0]
    commander = first[1].split(",")[0]
    agent = first[1].split(",")[1]
    cpc = first[1].split(",")[2]
    cpc_args = first[2].split(")")[0].split(",")
    cpc_full = cpc+"("+",".join(cpc_args)+")"
    second = text.split("{")[1].split("}")[0].replace(" ", "")
    supp_all = [i+")" for i in second.split("),")[:-1]]

    deconstructed = {"intent": intent.strip(),
            "central_proposition": cpc_full.strip(),
            "supplemental_semantics": supp_all}
    return deconstructed


