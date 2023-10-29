# Seismic Data

## Waveform Data

The seismic can be delivered in MiniSeed or any other format compatible with Pyrocko.

Organize your data in an [SDS structure](https://www.seiscomp.de/doc/base/concepts/waveformarchives.html) or just a blob if MiniSeed.

## Meta Data

Meta data is required primarily for station locations and codes.

Supported data formats are:

* [x] [StationXML](https://www.fdsn.org/xml/station/)
* [x] [Pyrocko Station YAML](https://pyrocko.org/docs/current/formats/yaml.html)

Metadata does not need to include response information for pure detection and localisation. If local magnitudes $M_L$ are extracted response information is required.
