"""CoIS method components; see README.md for release and reproduction scope."""
from .objective import cois_loss
from .context_views import source_context_probabilities
from .losses.ecoc import decode

__all__ = ["cois_loss", "source_context_probabilities", "decode"]
