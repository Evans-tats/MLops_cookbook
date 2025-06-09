#!/usr/bin/env python
import click
from mlib import predict

@click.command()
@click.option(
    '--weight',
    prompt='Enter weight in pounds',
    type=float,
    help='Weight to predict height of player'
)

def predictcle(weight):
    """Predicts height based on weight"""
    try:
        result = predict(weight)
        click.echo(f"Predicted height: {result['height_readable']} (inches: {result['height_inches']})")
    except Exception as e:
        click.echo(f"Error during prediction: {e}")

if __name__ == '__main__':
    predictcle()