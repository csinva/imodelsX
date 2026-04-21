"""Generate assets/blog_anim.gif — messy text on a conveyor belt goes through a
magnifying glass and emerges as a simple explanation on the other side."""

import os
import random

from PIL import Image, ImageDraw, ImageFont

W, H = 640, 220
BELT_TOP, BELT_BOT = 140, 170
LENS_CX, LENS_CY, LENS_R = W // 2, 110, 46
LEFT_EDGE, RIGHT_EDGE = 40, W - 40
N_FRAMES = 50
OUT = os.path.join(os.path.dirname(__file__), "assets", "blog_anim.gif")

random.seed(0)

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load(path, size):
    return ImageFont.truetype(path, size)


messy_font = load(MONO, 11)
clean_font = load(SANS, 20)

# messy text tokens — a wall of characters/snippets
vocab = [
    "lorem", "ipsum", "dolor", "sit", "amet", "xk92", "{42}", "??", "!!!",
    "the", "quick", "brown", "fox", "42", "q=0.7", "NaN", "<tok>", "id:7f",
    "0xAF", "null", "@@", "\\n", "<<", ">>", "u/17", "$$", "^^", "vec[3]",
    "bigram", "logit", "3.14", "0.001", "token", "word", "char", "embed",
    "raw", "noisy", "dense", "sparse", "hash", "0b101", "v2.1", "—",
]


def messy_blob():
    """Return a list of (dx, dy, text) offsets for one 'packet' of messy text."""
    blob = []
    for row in range(4):
        x = 0
        while x < 180:
            w = random.choice(vocab)
            blob.append((x, row * 12, w))
            x += len(w) * 7 + random.randint(2, 8)
    return blob


PACKETS = [messy_blob() for _ in range(3)]
# each packet starts offscreen-left and moves right; staggered
PACKET_STARTS = [-200, -440, -680]
CLEAN_TEXT = "✓"  # checkmark


def draw_belt(d):
    # belt body
    d.rectangle([LEFT_EDGE - 20, BELT_TOP, RIGHT_EDGE + 20, BELT_BOT], fill=(70, 70, 75))
    # top/bottom strips
    d.rectangle([LEFT_EDGE - 20, BELT_TOP, RIGHT_EDGE + 20, BELT_TOP + 4], fill=(40, 40, 45))
    d.rectangle([LEFT_EDGE - 20, BELT_BOT - 4, RIGHT_EDGE + 20, BELT_BOT], fill=(40, 40, 45))


def draw_rollers(d, phase):
    for cx in (LEFT_EDGE - 10, RIGHT_EDGE + 10):
        d.ellipse([cx - 14, BELT_TOP - 4, cx + 14, BELT_BOT + 4], fill=(110, 110, 115), outline=(30, 30, 30))
        # spoke to show rotation
        import math
        a = phase
        x2 = cx + 10 * math.cos(a)
        y2 = (BELT_TOP + BELT_BOT) / 2 + 10 * math.sin(a)
        d.line([cx, (BELT_TOP + BELT_BOT) / 2, x2, y2], fill=(30, 30, 30), width=2)


def draw_belt_texture(d, offset):
    # hash marks moving right to suggest motion
    y = (BELT_TOP + BELT_BOT) // 2
    step = 18
    x0 = LEFT_EDGE - 20 + (offset % step)
    x = x0
    while x < RIGHT_EDGE + 20:
        d.line([x, y - 5, x + 6, y + 5], fill=(95, 95, 100), width=1)
        x += step


def draw_magnifier(d):
    # handle
    import math
    hx1 = LENS_CX + int(LENS_R * math.cos(math.radians(45)))
    hy1 = LENS_CY + int(LENS_R * math.sin(math.radians(45)))
    hx2 = hx1 + 28
    hy2 = hy1 + 28
    d.line([hx1, hy1, hx2, hy2], fill=(60, 45, 30), width=8)
    d.ellipse([hx2 - 6, hy2 - 6, hx2 + 10, hy2 + 10], fill=(60, 45, 30))
    # rim
    d.ellipse(
        [LENS_CX - LENS_R, LENS_CY - LENS_R, LENS_CX + LENS_R, LENS_CY + LENS_R],
        outline=(35, 35, 40), width=6,
    )


def draw_lens_glass(overlay):
    # translucent glass
    od = ImageDraw.Draw(overlay)
    od.ellipse(
        [LENS_CX - LENS_R + 4, LENS_CY - LENS_R + 4, LENS_CX + LENS_R - 4, LENS_CY + LENS_R - 4],
        fill=(170, 210, 240, 90),
    )
    # glint
    od.ellipse([LENS_CX - 30, LENS_CY - 30, LENS_CX - 10, LENS_CY - 15], fill=(255, 255, 255, 120))


def frame(i):
    img = Image.new("RGB", (W, H), (245, 246, 248))
    d = ImageDraw.Draw(img)

    draw_belt(d)
    draw_belt_texture(d, -i * 6)
    draw_rollers(d, i * 0.5)

    # progress of animation [0, 1)
    t = i / N_FRAMES

    # draw messy packets moving right
    packet_w = 180
    speed = 28
    # frame at which the last packet's rightmost edge crosses the lens center
    last_in_lens_frame = (LENS_CX - PACKET_STARTS[-1] - packet_w) / speed
    for pi, packet in enumerate(PACKETS):
        px = PACKET_STARTS[pi] + speed * i
        # only draw letters whose x is left of the lens center (they disappear past it)
        for dx, dy, word in packet:
            x = px + dx
            y = BELT_TOP - 42 + dy
            # fade out as it approaches the lens
            if x < LENS_CX - LENS_R - 4:
                d.text((x, y), word, fill=(60, 60, 65), font=messy_font)
            elif x < LENS_CX:
                # inside the lens — blur-ish shrink effect by not drawing
                pass

    # clean text emerges only once every packet has fully entered the lens
    emergence_start = (last_in_lens_frame + 1) / N_FRAMES
    if t > emergence_start:
        # travels from just right of the lens and settles before the right edge
        local = (t - emergence_start) / (1 - emergence_start)
        start_x = LENS_CX + LENS_R - 10
        end_x = RIGHT_EDGE - 130
        cx = start_x + (end_x - start_x) * min(1.0, local * 1.6)
        ft = load(SANS, 32)
        bbox = d.textbbox((0, 0), CLEAN_TEXT, font=ft)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = cx - tw / 2
        ty = BELT_TOP - 14 - th
        # soft background pill
        pad = 6
        d.rounded_rectangle(
            [tx - pad, ty - pad, tx + tw + pad, ty + th + pad],
            radius=8, fill=(230, 245, 235), outline=(90, 160, 120), width=2,
        )
        d.text((tx, ty), CLEAN_TEXT, fill=(20, 100, 60), font=ft)

    # magnifier on top (with translucent glass)
    glass = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_lens_glass(glass)
    img = Image.alpha_composite(img.convert("RGBA"), glass).convert("RGB")
    d = ImageDraw.Draw(img)
    draw_magnifier(d)

    return img


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames = [frame(i) for i in range(N_FRAMES)]
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)")


if __name__ == "__main__":
    main()
