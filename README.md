# Unofficial implementation of MosaicSDF representation (no training yet)

This is an unofficial implementation of the MosaicSDF representation from the paper ["Mosaic-SDF for 3D Generative Models
"](https://lioryariv.github.io/msdf/)

The code includes functions for `mesh -> msdf` and `msdf -> mesh` conversion. The MSDF representation consists of a set of cubic grids parameterized by their center and scale. Each grid cell contains a signed distance value. For a more detailed explanation, please refer to the paper.



## MSDF representation cons

MSDF is a constant size representation for 3D shapes, so it can be easily used for training ML models (especially generative ones).

The representation's shape is `N x (K**3 + 3 + 1)`, where `N` is the number of grids (1024 in the paper and this repository), `K` is the grid size (7 in the paper and this repository), `3` accounts for the grid center, and `1` for the grid scale. Therefore, with the default parameters, a mesh can be represented as a `1024x347` tensor.

## Installation

Run `poetry install` to install all dependencies.

## Example of work

Example of work on test mesh from `assets/octopus/model.obj`:

```bash
python3  scripts/optimize_msdf.py
```

Original mesh ![original](assets/images/original.png)
Mesh after applying watertight preprocessing ![watertight](assets/images/watertight.png)
Sampled grid points ![grid](assets/images/mosaics.png)
Mesh reconstructed from msdf representation using marching cubes and resolution=32 ![reconstructed](assets/images/reconstruction.png)
Mesh after optimization ![optimized](assets/images/optimized.png)

## Implementation notes

### Speed

Despite efforts to enhance performance, mesh reconstruction from the MSDF representation remains slow, as it requires calculating the aggregation of all grid cells for each point within the grids. Optimization may be possible by focusing on the aggregation of only the top-K nearest grids.

### Optimization

There is a significant likelihood that there are issues with my optimization approach; it fails to converge, resulting in the optimized mesh appearing similar to the non-optimized version.

