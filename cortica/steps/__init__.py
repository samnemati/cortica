"""Concrete analysis steps (MNE-backed).

Each module registers one or more Step subclasses. MNE is imported lazily inside
``run()`` so that importing this package never requires MNE to be installed.
"""
