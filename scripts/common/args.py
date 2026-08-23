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


def add_ee_argument(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-ee",
        "--ee",
        dest="ee",
        action="store_true",
        help="Use the ee_ (end-effector) agent config variants instead of the "
        "originals. Selection (--scene/--tag) still matches the original tags.",
    )
