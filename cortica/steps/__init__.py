"""Concrete analysis steps (MNE-backed).

Each module registers one or more Step subclasses in the default registry. MNE is
imported lazily inside ``run()`` so that importing this package never requires MNE.
Importing this package imports every step module so they self-register.
"""
from . import epoch, fnirs, ica, montage, preprocess  # noqa: F401  (registration side effects)
