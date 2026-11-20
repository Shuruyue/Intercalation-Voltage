"""Average intercalation voltage of electrode materials from DFT, a machine-learned potential and gradient boosting,
and how far a model trained on lithium electrodes carries to sodium.

Modules are imported on use (``from intercalation_voltage import learning``) so that the GPAW environment, which has
no scikit-learn or matminer, can still import ``intercalation_voltage.structures``.
"""

__version__ = "0.1.0"
