import dask.array as da
import zarr
import requests
import io
from ome_zarr.io import parse_url
import ome_types
from vitessce import (
    VitessceConfig,
    CoordinationLevel as CL,
    get_initial_coordination_scope_prefix,
)

def load_and_analyze_data():
    # Load data
    path = "https://lsp-public-data.s3.amazonaws.com/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ/0"
    root = parse_url(path, mode="w")
    store = root.store
    daskArray = da.from_zarr(store, component="3")  # t=0, high-res
    print("Dask Array shape:", daskArray.shape)

    # Get metadata
    response = requests.get("https://lsp-public-data.s3.amazonaws.com/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ/OME/METADATA.ome.xml")
    data = response.text
    ome_xml = ome_types.from_xml(response.text.replace("Â",""))
    channel_names = [c.name for c in ome_xml.images[0].pixels.channels]
    print("\nChannel names:", channel_names)

    return create_vitessce_config()

def create_vitessce_config():
    # Create Vitessce configuration
    path = "https://lsp-public-data.s3.amazonaws.com/yapp-2023-3d-melanoma/Dataset1-LSP13626-melanoma-in-situ/0"
    vc = VitessceConfig(schema_version="1.0.16", name="BioMedVis Challenge")
    dataset = vc.add_dataset(name="Blood Vessel", uid="bv").add_file(
        url=path,
        file_type="image.ome-zarr"
    )

    spatial = vc.add_view("spatialBeta", dataset=dataset)
    lc = vc.add_view("layerControllerBeta", dataset=dataset)

    vc.link_views_by_dict([spatial, lc], {
        "spatialTargetZ": 0,
        "spatialTargetT": 0,
        "spatialZoom": -1.1,
        "spatialTargetX": 2914,
        "spatialTargetY": 1267,
        "spatialTargetZ": 0,
        "spatialRenderingMode": "3D",
        "imageLayer": CL([
            {
                "spatialTargetResolution": 3,
                "spatialLayerOpacity": 1.0,
                "spatialLayerVisible": True,
                "photometricInterpretation": "BlackIsZero",
                "imageChannel": CL([
                    {
                        "spatialTargetC": 9,
                        "spatialChannelColor": [255, 125, 0],
                        "spatialChannelVisible": True,
                        "spatialChannelOpacity": 1.0,
                        "spatialChannelWindow": [0,9486]
                    },
                    {
                        "spatialTargetC": 19,
                        "spatialChannelColor": [0, 255, 0],
                        "spatialChannelVisible": True,
                        "spatialChannelOpacity": 1.0,
                        "spatialChannelWindow": [666,21313]
                    },
                    {
                        "spatialTargetC": 69,
                        "spatialChannelColor": [255, 255, 0],
                        "spatialChannelVisible": True,
                        "spatialChannelOpacity": 1.0,
                        "spatialChannelWindow": [6,34]
                    },
                    {
                        "spatialTargetC": 21,
                        "spatialChannelColor": [255, 0, 0],
                        "spatialChannelVisible": True,
                        "spatialChannelOpacity": 1.0,
                        "spatialChannelWindow": [1638,36287]
                    },
                ])
            }
        ])
    }, meta=True, scope_prefix=get_initial_coordination_scope_prefix("bv", "image"))

    vc.layout(spatial | lc)
    return vc

if __name__ == "__main__":
    print("Loading and analyzing data...")
    vc = load_and_analyze_data() 