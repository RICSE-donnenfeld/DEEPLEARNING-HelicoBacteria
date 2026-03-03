"""
Backward-compatible wrapper.

Use `model_classifier.py` as canonical CNN script.
"""

from model_classifier import *  # noqa: F401,F403
from model_classifier import main


if __name__ == "__main__":
    main()


