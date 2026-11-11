# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""Printing a table, the only file that does."""

import logging

from deepska.glitch.kinds import Rows

log = logging.getLogger(__name__)


def table(title: str, header: str, rows: Rows, fmt: str) -> None:
    """
    Print a table.

    :param title: Line printed above the table.
    :param header: The column headings.
    :param rows: The rows, a tuple of values each.
    :param fmt: A % format for one row.
    """
    log.info(title)
    log.info(header)
    for row in rows:
        log.info(fmt, *row)
