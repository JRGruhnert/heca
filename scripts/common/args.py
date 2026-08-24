"""Shared argparse helpers for the pipeline scripts."""

import argparse


def add_scene_argument(parser: argparse.ArgumentParser, default=None):
    parser.add_argument(
        "--scene",
        default=default,
        help="Scene module tag (e.g. scene1, sceneog).",
    )


def add_tag_argument(parser: argparse.ArgumentParser, default=None):
    parser.add_argument(
        "--tag",
        default=default,
        help="Agent tag within the selected scene.",
    )


def add_model_argument(parser: argparse.ArgumentParser, default=None):
    parser.add_argument(
        "--model",
        default=default,
        help="Agent tag within the selected scene.",
    )
