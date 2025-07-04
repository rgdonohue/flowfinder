#!/usr/bin/env python3
"""
FLOWFINDER Command Line Interface Entry Point
============================================

This module provides the entry point for running FLOWFINDER as a module:
    python -m flowfinder

It delegates to the main CLI functionality in cli.py
"""

from .cli import main

if __name__ == "__main__":
    main()
