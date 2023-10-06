#!/usr/bin/env python
#evaluate.py
import sys
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import json
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.isomorphism as iso
import ast

# Various models that I have trained (peft_model_id, base_model)
PARSERS = [("vsarathy/parser-info_structure-30k-no-context-llama2-7b", "codellama/CodeLlama-7b-hf"),
           ("vsarathy/parser-info_structure-30k-no-context-llama2-3b", "openlm-research/open_llama_3b_v2"),
           ("vsarathy/parser-info_structure-30k-no-context-falcon-7b", "tiiuae/falcon-7b")]
PARSERS_CONTEXT = [("vsarathy/parser-info_structure-30k-context-llama2-7b", "codellama/CodeLlama-7b-hf"),
                   ("vsarathy/parser-info_structure-30k-context-llama2-3b", "openlm-research/open_llama_3b_v2"),
                   ("vsarathy/parser-info_structure-30k-context-falcon-7b", "tiiuae/falcon-7b")]
TRANSLATORS = [("vsarathy/translator-simple_english-30k-no-context-llama2-7b", "codellama/CodeLlama-7b-hf"),
               ("vsarathy/translator-simple_english-30k-no-context-llama2-3b", "openlm-research/open_llama_3b_v2"),
               ("vsarathy/translator-simple_english-30k-no-context-falcon-7b", "tiiuae/falcon-7b")]
TRANSLATORS_CONTEXT = [("vsarathy/translator-simple_english-30k-context-llama2-7b", "codellama/CodeLlama-7b-hf"),
               ("vsarathy/translator-simple_english-30k-context-llama2-3b", "openlm-research/open_llama_3b_v2"),
               ("vsarathy/translator-simple_english-30k-context-falcon-7b", "tiiuae/falcon-7b")]

PARSING_DATA = "vsarathy/nl-robotics-semantic-parsing-info_structure-10k-no-context-TEST"
PARSING_CONTEXT_DATA = "vsarathy/nl-robotics-semantic-parsing-info_structure-10k-context-TEST"
TRANSLATION_DATA = "vsarathy/nl-robotics-translation-simple_english-12k-no-context-TEST"
TRANSLATION_CONTEXT_DATA = "vsarathy/nl-robotics-translation-simple_english-12k-context-TEST"


def main(args):
    # Load dataset
    print("\n>> LOADING DATASET")
    if not args.context:
        if "par" in args.type.lower():
            model_names = PARSERS
            dataset_name = PARSING_DATA
        else:
            model_names = TRANSLATORS
            dataset_name = TRANSLATION_DATA
    else:
        if "par" in args.type.lower():
            model_names = PARSERS_CONTEXT
            dataset_name = PARSING_CONTEXT_DATA
        else:
            model_names = TRANSLATORS_CONTEXT
            dataset_name = TRANSLATION_CONTEXT_DATA
    dataset = load_dataset(dataset_name, split="test")
    print(f"\tLoaded models: {model_names}\ndataset: {dataset_name}")

    # Running Inference
    print(">> RUNNING INFERENCE")
    for peft_model_id, base_model_id in model_names:
        print(f"\n>> Loading models for:\nPEFT: {peft_model_id}")
        predictions = []
        model, tokenizer = load_peft_model(peft_model_id, base_model_id)
        print(">> Performing inference")
        for item in tqdm(dataset, position=0, leave=False, desc="Data Items"):
            print("\t Generating output")
            prompt = f"{item['instruction']}\n\nutterance:\n{item['input']}\n\nJSON:\n"
            output = generate(prompt,model, tokenizer)
            if args.verbose:
                print(f"\t Input: {item['input']}\nGenerated: {output}")
            item['predicted'] = output
            item['model'] = peft_model_id
            item['base_model'] = base_model_id
            print("\t Evaluating output")
            item['evaluation'] = evaluate(item['predicted'], item['output'])
            predictions.append(item)

        # Save results to file 
        print(f"Model ({peft_model_id}) evaluation completed")
        filename = f"{peft_model_id}_evaluation.json"
        with open(filename, "w") as f:
            json.dump(predictions, f)
        print(f"Results saved in {filename}")

def load_peft_model(peft_model_id, base_model_id):
  # Load the PEFT adapter into a model
  model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="auto",load_in_4bit=True)

  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained(base_model_id)
  return model, tokenizer

def generate(prompt, model, tokenizer, params={'max_new_tokens': 100}):
  # Run the model
  model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
  generated_ids = model.generate(**model_inputs, max_new_tokens=params['max_new_tokens'])
  results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
  output = results.split("\n")[-1]
  return output

def evaluate(predicted, truth):
    """
    Returns an evaluation json that contains various metrics 
    input: strings obtained from the language model
    input: json of the ground truth
    """
    item = {}
    item['truth'] = truth
    if isinstance(predicted, str):
        # Validate json 
        predicted_json = {}
        try:
            predicted_json = ast.literal_eval(predicted_json_str)
            item['valid_json'] = True
            item['prediction'] = predicted_json
        except:
            item['valid_json'] = False
            pass
    
        # try to fix the json with gpt
        if not item['valid_json']:
            # try to fix it.
            new_json_str = fix_json(predicted)
            
            try: 
                predicted_json = ast.literal_eval(new_json_str)
                item['prediction'] = predicted_json
            except e:
                print("Unable to generate a valid dictionary.")
                return item
    else:
        item['prediction'] = predicted
        item['valid_json'] = True

    print("\t>> Checking Intent")
    # check intent
    if item['prediction']['intent'] == truth['intent']:
        item['intent_correct'] = True
    else:
        item['intent_correct'] = False

    # check cpc name
    if  is_same_pred_name(item['prediction']['central_proposition'], truth['central_proposition']):
        item['cpc_name_correct'] = True
    else:
        item['cpc_name_correct'] = False

    # check if correct number of sups
    spc_name_prediction = [pred_name(i) for i in item['prediction']['supplemental_semantics']]
    spc_name_truth = [pred_name(i) for i in truth['supplemental_semantics']]

    if len(spc_name_prediction) == len(spc_name_truth):
        item['spc_length_correct'] = True
    else:
        item['spc_length_correct'] = False

    # Evaluate accuracy of spc
    item['spc_accuracy'] = {}
    spc_intersection = set(spc_name_prediction).intersection(set(spc_name_truth))
    item['spc_accuracy']['precision'] = len(spc_intersection)/len(spc_name_prediction)
    item['spc_accuracy']['recall'] = len(spc_intersection)/len(spc_name_truth)

    # check for variable assignment and mapping. 
    if is_isomorphic(item['prediction'],truth):
        item['is_isomorphic'] = True
    else:
        item['is_isomorphic'] = False

    if is_matched(item['prediction'], truth):
        item['is_matched'] = True
    else:
        item['is_matched'] = False

    return item

#### graph matching

def build_semantic_graph(parse):
    """
    builds up a graph 
    """

    G = nx.DiGraph()

    # let's first do the "intent"
    cpc_name = pred_name(parse['central_proposition'])
    cpc_args = pred_args(parse['central_proposition'])
    
    G.add_node(cpc_name, name=cpc_name, source='cpc', type='pred_name')
    for idx,arg in enumerate(cpc_args):
        G.add_node(arg, name=arg, source='args', type='pred_arg')
        G.add_edge(arg,cpc_name,pos=idx)

    for spc in parse['supplemental_semantics']:
        spc_name = pred_name(spc)
        spc_args = pred_args(spc)
        G.add_node(spc_name, name=spc_name, source='spc', type='pred_name')
        for idx, arg in enumerate(spc_args):
            G.add_node(arg, name=arg, source='args', type='pred_arg')
            G.add_edge(arg,spc_name,pos=idx)
    return G

def is_isomorphic(predicted, truth):
    """
    Checks if all the right variables are positioned correctly in the CPC and SPC
    """
    G_predicted = build_semantic_graph(predicted)
    G_truth = build_semantic_graph(truth)
    em = iso.categorical_edge_match("pos", 1)
    return nx.is_isomorphic(G_truth, G_predicted, edge_match=em)

def is_matched(predicted, truth):
    """
    Checks if each variable is correctly connected to exactly the same set of cpc and spcs. 
    """
    G_predicted = build_semantic_graph(predicted)
    G_truth = build_semantic_graph(truth)

    # get all the nodes that are variables
    args_predicted = [x for x,y in G_predicted.nodes(data=True) if y['type']=='pred_arg']
    args_truth = [x for x,y in G_truth.nodes(data=True) if y['type']=='pred_arg']

    for p,t in zip(args_predicted, args_truth):
        successors_predicted = G_predicted.successors(p)
        successors_truth = G_truth.successors(t)
        s1 = set(successors_predicted)
        s2 = set(successors_truth)
        if not s1 == s2: 
            return False
    return True
    

### UTILITIES ####

def fix_json(json_str):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)
    template = """
        Fix the input json string to produce an output that has a valid json format. Only change things like the parenthesis, commas etc.
        
        json_input: \n{json_str}\n
        rewritten valid json:
        """
    prompt = PromptTemplate(input_variables=["json_str"],template=template)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run(json_str=json_str)
    return output

def is_same_pred_name(predicted_pred, truth_pred):
    if predicted_pred.split("(")[0] == truth_pred.split("(")[0]:
        return True
    return False

def pred_name(pred):
    return pred.split("(")[0].lower()

def pred_args(pred):
    return pred.split("(")[1].split(")")[0].split(",")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default = "parser", required=True)
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity") 
    parser.add_argument("--context", action="store_true", help="if set, chooses context") 
    args = parser.parse_args()
    main(args)

