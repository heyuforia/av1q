"""Encoder back-ends. One module per engine; the Grid/Engine interface
they implement lives in base.py. The shared brain must only ever import
from base — concrete engines are selected by the launchers."""
