import os
import click
import json
import random
import ast
from gsp import pipelines
from gsp import preparation

# Shared click options
shared_options = [
    click.option('--verbose/--no-verbose', '-v', default=False, help="If set, console output is verbose"),
]

def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options

@click.group()
@click.option('--verbose/--no-verbose', '-v', default=False, help="If set, console output is verbose")
@click.pass_context
def cli(ctx, **kwargs):
    ctx.ensure_object(dict)
    ctx.obj = kwargs
    click.clear()
    click.secho('Grounded Semantic Parser', bold=True, fg='blue')
    click.secho(f"GSP: Grounded Semantic Parser", fg='yellow')
    print(f'-----------------')


@click.command()
@add_options(shared_options)
@click.pass_context
def serve(ctx, **kwargs):
    from gsp import websvc
    ctx.obj.update(kwargs)
    websvc.main()


@click.command()
@click.option('--input', '-i', default="", help="Provide input json file path")
@click.option('--model', '-m', default="gpt-4", help="Provide LLM model name")
@click.option('--style', '-s', 
              default=['directness', 'familiarity', 'formality', 'disfluency', 'word_choice', 'none'], 
              help="Provide a list of possible styles to be applied")
@click.option('--num-variations', '-r', default=5, help="Provide the number of style variations generated for each style")
@click.option('--num-data', '-n', default=5, help="Provide the number of data items to be pulled from input")
@click.option('--wav', '-w',
              default = ["crop", "mask", "noise", "pitch", "speed", "normalize", "polarity_inversion", "none"],
              help="Provide a list of possible audio/wave augmentations to be applied")
@click.option('--accent', '-a',
              default = ["indian", "american", "irish", "australian", "none"],
              help="Provide a list of possible accent augmentations to be applied")
@click.option('--text', '-t',
              default=["back_translation", "synonym", "span_crop", "contextual_embedding", "none"],
              help="Provide a list of possible text augmentations to be applied")
@add_options(shared_options)
@click.pass_context
def style_augment(ctx, **kwargs):
    ctx.obj.update(kwargs)
    click.secho("\nRunning Data Augmentation Pipeline\n", bold=True, fg='white')

    ctx.obj['style'] = ast.literal_eval(ctx.obj['style'])
    ctx.obj['accent'] = ast.literal_eval(ctx.obj['accent'])
    ctx.obj['wav'] = ast.literal_eval(ctx.obj['wav'])
    ctx.obj['text'] = ast.literal_eval(ctx.obj['text'])

    from gsp import pipelines
    pipelines.run(ctx.obj)


@click.command()
@click.option('--input', '-i', default="", help="Provide input json file path")
@click.option('--model', '-m', default="gpt-4", help="Provide LLM model name")
@click.option('--num-data', '-n', default=5, help="Provide the number of data items to be pulled from input")
@click.option('--num-variations', '-r', default=5, help="Provide the number of style variations generated for each style")
@click.option('--style', '-s', 
              default=['directness', 'familiarity', 'formality', 'disfluency', 'word_choice', 'asr', 'correction', 'none'], 
              help="Provide a list of possible styles to be applied")
@add_options(shared_options)
@click.pass_context
def stylize(ctx, **kwargs):
    ctx.obj.update(kwargs)
    click.secho("\nRunning Stylizing Pipeline\n", bold=True, fg='green')
    ctx.obj['style'] = ast.literal_eval(ctx.obj['style'])
    pipelines.stylize(ctx.obj)


@click.command()
@click.option('--input', '-i', default="", help="Provide input json file path")
@click.option('--num-data', '-n', default=5, help="Provide the number of data items to be pulled from input")
@click.option('--wav/--no-wave', '-w',
              default=False,
              help="If set, will augment with audio interruptions.")
@click.option('--accent/--no-accent', '-a',
              default=False,
              help="If set, will augment with accents")
@click.option('--text/--no-text', '-t',
              default=False,
              help="If set, will augment with text variations")
@click.option('--language/--no-language', '-l',
              default=False,
              help="If set, will augment with different language translations")
@click.option('--preaccelerate/--no-preaccelerate', '-p',
              default=False,
              help="If set, will activate preacceleration for language translation tasks (back_translation and translation)")
@add_options(shared_options)
@click.pass_context
def augment(ctx, **kwargs):
    ctx.obj.update(kwargs)
    click.secho("\nRunning Augmentation Pipeline\n", bold=True, fg='green')
    pipelines.augment(ctx.obj)



@click.command()
@click.option('--input', '-i', default=[], multiple=True, help="Provide using repeated -i <FILENAME> -i <FILENAME> ...")
@add_options(shared_options)
@click.pass_context
def merge(ctx, **kwargs):
    ctx.obj.update(kwargs)
    click.secho("\nMerge files\n", bold=True, fg='green')
    pipelines.merge(ctx.obj)



@click.command()
@click.option('--input', '-i', default="", help="Provide input json file path")
@click.option('--purpose', '-p', default="train", help="Is it a 'train' or 'test' dataset?")
@click.option('--context/--no-context', '-c',
              default=False,
              help="If set, the dataset will include the action/detection repertoire within the input and instructions of the prompt.")
@click.option('--simple/--no-simple', '-s',
              default=False,
              help="If set, the dataset will include as output not a json structure, but a simple english translation, which is basically the basic_utterance.")
@add_options(shared_options)
@click.pass_context
def prepare(ctx, **kwargs):
    ctx.obj.update(kwargs)
    click.secho("\nPreparing dataset for fine tuning\n", bold=True, fg="white")
    preparation.prepare(ctx.obj)


cli.add_command(serve)
cli.add_command(style_augment)
cli.add_command(prepare)
cli.add_command(stylize)
cli.add_command(augment)
cli.add_command(merge)

def main():
    cli(obj={})

if __name__ == "__main__":
    main()

