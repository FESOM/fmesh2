# Render spike (2026-07-10)

Validated: NG5 (7.4M nodes / 14.7M tri / 22M edges) at 60fps on M3 Max, globe<->Plate-Carree shader morph, dateline seam duplication, NE-10m coastline overlay.

Regenerate buffers and serve:

    micromamba run -n meshgen python convert.py ../../../core2 core2 core2
    micromamba run -n meshgen python convert.py ../../../NG5_mesh ng5 NG5
    curl -sLo ne_10m_coastline.geojson https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_coastline.geojson
    micromamba run -n meshgen python coast_convert.py ne_10m_coastline.geojson .
    micromamba run -n meshgen python -m http.server 8077
    # open http://127.0.0.1:8077/index.html?mesh=ng5
