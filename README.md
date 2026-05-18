# FLODAS

FLODAS is a UNET model for detecting floating matter-loaded windrows in coastal scenes on 8-band Planet SuperDove imagery.

The model has been trained on multiple scenes from the North Eastern Mediterranean (Aegean Sea and Sea of Marmara) containing various floating matter instances (post-flooding debris, 
marine mucilage, pollen, artificial floating targets).

The script runs full-scene tiled inference, creates a soft probability raster, applies configurable post-processing, and exports a final binary detection mask and vector products.

---

## Repository structure

The release folder should contain the following files:

```text
FLODAS/
├── FLODAS.py
├── model_unet.py
├── config.yaml
├── requirements.txt
├── unet_FLODAS.pth --> download from 
└── README.md
```
Files should be kept in the same folder.

User only needs to provide input raster path when running the script.

---

## Input data requirements

The input raster must be an 8-band Planet SuperDove GeoTIFF.

Expected band order:

```text
443, 490, 531, 565, 610, 665, 705, 865 nm
```

---

## Installation

A virtual environment is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU support, install the appropriate PyTorch version for your CUDA setup before installing the remaining dependencies from `requirements.txt`.

## Running FLODAS

From inside the FLODAS folder, run:

```powershell
python FLODAS.py C:\path\to\PlanetSuperDove_scene.tif
```

The script automatically looks for the following files in the same folder as `FLODAS.py`:

```text
model_unet.py
config.yaml
*.pth
```

If there is only one `.pth` file in the folder, it will be used automatically.

If there are multiple `.pth` files, set the desired filename in `config.yaml`.

---

## Configuration

Post-processing and tiling settings are controlled from `config.yaml`.

### Main settings

`threshold`  
Probability threshold used to convert the soft probability raster into a binary detection mask.

`min_object_size`  
Minimum object size in pixels. Smaller connected components are removed.

`edge_buffer_px`  
Distance in pixels excluded around valid-data boundaries. This helps remove unreliable predictions near raster edges and nodata boundaries.

`apply_gaussian`  
If `true`, Gaussian smoothing is applied to the probability raster before thresholding.

`gauss_sigma`  
Controls the strength of Gaussian smoothing.

`apply_morph_smoothing`  
If `true`, morphological closing is applied to the binary mask.

`morph_size`  
Size of the square structuring element used for morphological closing.

`fill_holes`  
If `true`, small holes inside detected regions are filled.

`min_hole_px`  
Maximum hole size, in pixels, to fill.

`isobands.enabled`  
If `true`, probability isobands are exported as vector polygons.

---

## Outputs

All outputs are written to the same folder as the input raster.

For an input raster named:

```text
raster.tif
```

the outputs will have names similar to:

```text
raster_unet_FLODAS_YYYYMMDD-HHMM_prob.tif
raster_unet_FLODAS_YYYYMMDD-HHMM_thr90.tif
raster_unet_FLODAS_YYYYMMDD-HHMM_thr90.shp
raster_unet_FLODAS_YYYYMMDD-HHMM_thr90.shp
raster_unet_FLODAS_YYYYMMDD-HHMM_isobands_10-100.shp
```

### Output descriptions

`*_prob.tif`  
Soft probability raster. Values range from 0 to 1.

`*_thrXX.tif`  
Final binary detection mask after thresholding and post-processing.

`*_thrXX.shp`  
Vectorized polygons of the final positive detections.

`*_excluded-*.shp`  
Vector polygons showing excluded regions, such as edge-buffer areas and removed small objects.

`*_isobands_10-100.shp`  
Vector probability bands, exported in 10 percent increments by default.

---

## Citation

Papageorgiou, D.; Aliani, S.; Cózar, A.; Topouzelis, K. SuperDove Satellite Detection of Matter-Loaded Windrows in the Aegean and Marmara Seas with a Focus on Extreme
Environmental Events. International Journal of Applied Earth Observation and Geoinformation (under review), 2026


