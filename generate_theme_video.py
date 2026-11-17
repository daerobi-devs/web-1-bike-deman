"""
Generator Video Animasi Bertema Bike Sharing Demand (MP4 H.264)
Menggunakan Pillow + imageio_ffmpeg tanpa perlu instalasi ffmpeg sistem eksternal.
"""
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540
FPS = 24
DURATION_SEC = 8
TOTAL_FRAMES = FPS * DURATION_SEC

# Kurva rata-rata penyewaan 24 jam (representatif pola komuter Bike Sharing)
HOURLY_DEMAND = [
    54, 33, 22, 11, 6, 19, 76, 213, 362, 221, 175, 206,
    248, 257, 243, 254, 316, 468, 430, 315, 228, 173, 133, 89
]


def load_font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf" if not bold else "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arial.ttf" if not bold else "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_frame(frame_idx: int) -> np.ndarray:
    progress = min(1.0, frame_idx / (TOTAL_FRAMES - 18))
    img = Image.new("RGB", (W, H), (15, 23, 42))  # #0f172a
    draw = ImageDraw.Draw(img)

    f_badge = load_font(14, bold=True)
    f_title = load_font(28, bold=True)
    f_sub = load_font(16, bold=False)
    f_kpi = load_font(30, bold=True)
    f_lbl = load_font(14, bold=False)

    # Top badge & header
    draw.rounded_rectangle([48, 34, 290, 62], radius=14, fill=(14, 116, 144))
    draw.text((64, 40), "KELOMPOK 7 UNPAM | ML TERAPAN", fill=(240, 249, 255), font=f_badge)

    draw.text((48, 76), "Simulasi Pola Penyewaan Sepeda 24 Jam", fill=(248, 250, 252), font=f_title)
    draw.text((48, 114), "Dataset: Bike Sharing Demand (10.886 Record) | Washington D.C.", fill=(148, 163, 184), font=f_sub)

    # Chart card container
    cx0, cy0, cx1, cy1 = 48, 155, 912, 435
    draw.rounded_rectangle([cx0, cy0, cx1, cy1], radius=12, fill=(30, 41, 59), outline=(51, 65, 85), width=2)

    # Grid lines
    gx0, gy0, gx1, gy1 = cx0 + 60, cy0 + 35, cx1 - 40, cy1 - 45
    max_val = 500
    for val in [0, 125, 250, 375, 500]:
        y = int(gy1 - (val / max_val) * (gy1 - gy0))
        draw.line([(gx0, y), (gx1, y)], fill=(51, 65, 85), width=1)
        draw.text((cx0 + 18, y - 8), str(val), fill=(148, 163, 184), font=f_lbl)

    # X-axis labels
    for h in range(0, 24, 3):
        x = int(gx0 + (h / 23.0) * (gx1 - gx0))
        draw.text((x - 14, gy1 + 10), f"{h:02d}:00", fill=(148, 163, 184), font=f_lbl)

    # Animated curve points
    exact_idx = progress * 23.0
    max_i = int(exact_idx)
    pts = []
    for i in range(max_i + 1):
        px = gx0 + (i / 23.0) * (gx1 - gx0)
        py = gy1 - (HOURLY_DEMAND[i] / max_val) * (gy1 - gy0)
        pts.append((px, py))

    if max_i < 23 and exact_idx > max_i:
        frac = exact_idx - max_i
        v_interp = HOURLY_DEMAND[max_i] * (1 - frac) + HOURLY_DEMAND[max_i + 1] * frac
        px = gx0 + (exact_idx / 23.0) * (gx1 - gx0)
        py = gy1 - (v_interp / max_val) * (gy1 - gy0)
        pts.append((px, py))
        curr_hour = int(round(exact_idx))
        curr_val = int(round(v_interp))
    else:
        curr_hour = max_i
        curr_val = HOURLY_DEMAND[max_i]

    if len(pts) >= 2:
        draw.line(pts, fill=(56, 189, 248), width=4)

    # Active dot & callout
    lx, ly = pts[-1]
    draw.ellipse([lx - 7, ly - 7, lx + 7, ly + 7], fill=(248, 250, 252), outline=(2, 132, 199), width=3)

    # Live KPI overlay inside top-right of chart
    draw.rounded_rectangle([650, 172, 892, 242], radius=8, fill=(15, 23, 42), outline=(2, 132, 199), width=1)
    draw.text((666, 180), f"Pukul {curr_hour:02d}:00 WIB", fill=(148, 163, 184), font=f_lbl)
    draw.text((666, 200), f"{curr_val} unit/jam", fill=(56, 189, 248), font=f_kpi)

    # Footer bar
    draw.text((48, 466), "Model Utama: HistGradientBoosting (Akurasi R2 = 95,23%)", fill=(226, 232, 240), font=f_sub)
    draw.text((48, 492), "Terhubung ke Daeroom Cloud PostgreSQL (db_bikesharingdemand)", fill=(148, 163, 184), font=f_lbl)

    return np.array(img)


def main() -> None:
    out_path = Path(__file__).resolve().parent / "assets" / "bike_sharing_demo.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out_path), fps=FPS, codec="libx264", pixelformat="yuv420p") as writer:
        for i in range(TOTAL_FRAMES):
            writer.append_data(render_frame(i))
    print(f"Video berhasil dibuat: {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
