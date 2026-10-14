from pathlib import Path
from PIL import Image
import tifffile

roots = [Path.home() / "Desktop", Path.cwd()]
hits = []
for root in roots:
    if root.exists():
        for p in root.rglob("*.tif"):
            if any(key in p.name for key in ["人口", "pop", "Pop", "chn_pop", "100m"]):
                hits.append(p)

for f in hits:
    print("\nFILE", f)
    print("size_bytes", f.stat().st_size)
    try:
        img = Image.open(f)
        print("PIL", img.size, img.mode)
    except Exception as exc:
        print("PIL_ERROR", repr(exc))
    try:
        with tifffile.TiffFile(str(f)) as tf:
            page = tf.pages[0]
            print("shape", page.shape, "dtype", page.dtype)
            keys = ["ModelPixelScaleTag", "ModelTiepointTag", "GeoKeyDirectoryTag", "GDAL_NODATA"]
            tags = {t.name: t.value for t in page.tags.values() if t.name in keys}
            print(tags)
    except Exception as exc:
        print("TIFF_ERROR", repr(exc))

