# Useful prompts

prompt_template_speech_act_theoretic = """
### instruction
Given an utterance, extract a semantic parse of the utterance and respond with the parse in a perfect JSON format. 

### example 
utterance: put the potted plant outside of the skis
JSON: 

{{
    "referents": [[
        {{
            "variable_name": "var0",
            "descriptors": [[
                "pottedplant"
                ]],
            "cognitive_status": "DEFINITE"
        }},
        {{
            "variable_name": "var1",
            "descriptors": [[
                "skis"
                ]],
            "cognitive_status": "DEFINITE"
        }}
    ]],
    "intent": "INSTRUCT",
    "central_propositional": {{ 
        "label": "putoutside",
        "type": "action",
        "arguments": [[
            "self:agent", 
            "VAR0", 
            "VAR1"
            ]]
    }},
}}


### utterance: {utterance}
### JSON:
{parse}
"""

instruction = "Given an utterance, extract a semantic parse of the utterance and respond with the parse in a perfect JSON format."
instruction_with_context = "Given an utterance and a context comprising a set of action and detection capabilities, extract a semantic parse of the utterance commensurate with the actions and detection abilities, and respond with the parse in a perfect JSON format."

example = """
Here is an example of a parse for an utterance. 
utterance:
put the potted plant outside of the skis

JSON:

{{
    "intent": "INSTRUCT",
    "central_proposition": "putoutside(self:agent,VAR0,VAR1)",
    "supplemental_semantics": [[ "pottedplant(VAR0)", "skis(VAR1)", "DEFINITE(VAR0)", "DEFINITE(VAR1)" ]]
}}
"""

example_with_context = """
Here is an example of a parse for an utterance. 
utterance:
put the potted plant outside of the skis

action capabilities:
{actions}

detection capabilities:
{properties}


JSON:

{{
    "intent": "INSTRUCT",
    "central_proposition": "putoutside(self:agent,VAR0,VAR1)",
    "supplemental_semantics": [[ "pottedplant(VAR0)", "skis(VAR1)", "DEFINITE(VAR0)", "DEFINITE(VAR1)" ]]
}}
"""


prompt_template_deconstructed_goal_semantics = """
### instruction
{instruction}

### example
{example}

### utterance
{input}

### JSON: 
{output}
"""


prompt_template_deconstructed_goal_semantics_with_context = """
### instruction
{instruction_with_context}

### example
{example_with_context}

### utterance
{utterance}

### actions
{actions}

### properties
{properties}

### JSON: 
{output}
"""
