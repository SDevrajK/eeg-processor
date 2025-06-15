import click
from src.eeg_processor.pipeline import EEGPipeline


@click.group()
def cli():
    pass


@cli.command()
@click.argument('config_path')
def batch(config_path):
    """Run batch processing"""
    pipeline = EEGPipeline(config_path)
    pipeline.run()


@cli.command()
@click.argument('config_path')
@click.argument('participant_id')
def explore(config_path, participant_id):
    """Launch interactive exploration"""
    pipeline = EEGPipeline(config_path)
    pipeline.load_participant(participant_id)

    # Start IPython shell for interactive exploration
    import IPython
    IPython.embed()


if __name__ == '__main__':
    cli()