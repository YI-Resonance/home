import os
import traceback
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. 基本參數設定
# ==========================================
FONT_FILENAME = "simli.ttf"
CSV_FILENAME = "曼陀羅.csv"

TEXT_STRETCH_Y = 1.15
FONT_SIZE = 465

BASE_MARGIN = 120

FRAME_ALPHA = 77
FRAME_WIDTH_BASE = 20

COLOR_MAP = {
    '入': '#EE3333',
    '陽平': '#1151F7',
    '陰平': '#33EEEE',
    '仄': '#FF970D',
    '拗平': '#B900FF',
    '拗仄': '#FF00B9',
    '破': '#5A2E1A'
}

# ==========================================
# 2. 顏色輔助函式
# ==========================================
def hex_to_rgba(hex_str, alpha=255):
    hex_str = hex_str.lstrip('#')
    r, g, b = [int(hex_str[i:i+2], 16) for i in (0, 2, 4)]
    return (r, g, b, alpha)

def darken_color(hex_str, factor=0.45, alpha=255):
    """將背景色加深，factor 越小越深"""
    hex_str = hex_str.lstrip('#')
    r, g, b = [int(hex_str[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return (r, g, b, alpha)

# ==========================================
# 3. 文字圖層
# ==========================================
def create_stretched_text_layer(text, font, text_color, final_size, stretch_y, is_rhyme=False):
    temp_size = 500
    temp_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)

    temp_draw.text((50, 50), text, font=font, fill=text_color)
    bbox = temp_draw.textbbox((50, 50), text, font=font)

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return Image.new('RGBA', (final_size, final_size), (0, 0, 0, 0))

    char_img = temp_img.crop(bbox)
    orig_w, orig_h = char_img.size
    new_h = int(orig_h * stretch_y)
    char_img_stretched = char_img.resize((orig_w, new_h), Image.Resampling.LANCZOS)

    text_layer = Image.new('RGBA', (final_size, final_size), (0, 0, 0, 0))
    text_x = (final_size - orig_w) // 2
    text_y = (final_size - new_h) // 2
    text_layer.paste(char_img_stretched, (text_x, text_y), char_img_stretched)

    if is_rhyme:
        draw_text_layer = ImageDraw.Draw(text_layer)
        line_y = text_y + new_h + 30
        line_w = int(orig_w * 0.95)
        line_x1 = (final_size - line_w) // 2
        line_x2 = line_x1 + line_w
        draw_text_layer.line([(line_x1, line_y), (line_x2, line_y)], fill=(0, 0, 144, 255), width=20)

    return text_layer

# ==========================================
# 4. 幾何框
# ==========================================
def draw_dashed_ellipse(draw, bbox, outline, width, dash_len=15, space_len=10):
    x0, y0, x1, y1 = bbox
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    circ = np.pi * (3 * (rx + ry) - np.sqrt((3 * rx + ry) * (rx + 3 * ry)))
    num_dashes = max(1, int(circ / (dash_len + space_len)))
    angle_step = 360 / num_dashes
    dash_angle = angle_step * (dash_len / (dash_len + space_len))
    for i in range(num_dashes):
        start = i * angle_step
        end = start + dash_angle
        draw.arc(bbox, start=start, end=end, fill=outline, width=width)

def draw_verb_arrow(draw, cx, cy, r, angle_deg, pointing_clockwise, frame_color, scale=1):
    arrow_len = 20 * scale
    arrow_width = 12 * scale

    target_rad = np.radians(angle_deg)
    ex = cx + r * np.cos(target_rad)
    ey = cy + r * np.sin(target_rad)

    tangent_rad = np.radians(angle_deg + 90) if pointing_clockwise else np.radians(angle_deg - 90)

    tip_x = ex + 2 * scale * np.cos(tangent_rad)
    tip_y = ey + 2 * scale * np.sin(tangent_rad)
    back_x = tip_x - arrow_len * np.cos(tangent_rad)
    back_y = tip_y - arrow_len * np.sin(tangent_rad)

    p1x = back_x + arrow_width * np.cos(tangent_rad + np.pi/2)
    p1y = back_y + arrow_width * np.sin(tangent_rad + np.pi/2)
    p2x = back_x + arrow_width * np.cos(tangent_rad - np.pi/2)
    p2y = back_y + arrow_width * np.sin(tangent_rad - np.pi/2)

    draw.polygon([(tip_x, tip_y), (p1x, p1y), (p2x, p2y)], fill=frame_color)

def generate_supersampled_frame(pos, line_idx=1, base_margin=45, final_size=300, scale=4, frame_color=None):
    size = final_size * scale
    cx, cy = size // 2, size // 2

    m = base_margin * scale
    width = FRAME_WIDTH_BASE * scale

    hi_res_img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(hi_res_img)

    if pos == '名':
        draw.rectangle((m, m, size - m, size - m), outline=frame_color, width=width)

    elif pos == '動':
        r = (size - 2 * m) / 2
        is_odd = (line_idx % 2 != 0)

        if is_odd:
            draw.arc((m, m, size - m, size - m), start=210, end=150, fill=frame_color, width=width)
            draw_verb_arrow(draw, cx, cy, r, angle_deg=160, pointing_clockwise=True, frame_color=frame_color, scale=FRAME_WIDTH_BASE*0.7)
        else:
            draw.arc((m, m, size - m, size - m), start=30, end=330, fill=frame_color, width=width)
            draw_verb_arrow(draw, cx, cy, r, angle_deg=20, pointing_clockwise=False, frame_color=frame_color, scale=FRAME_WIDTH_BASE*0.7)

    elif pos in ['形', '副', '形容', '副詞', '形容詞']:
        r = cx - m
        poly = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        draw.polygon(poly, outline=frame_color, width=width)

        s = 15 * scale
        draw.line([(cx, cy - r - s), (cx, cy - r + s)], fill=frame_color, width=2 * scale)
        draw.line([(cx + r - s, cy), (cx + r + s, cy)], fill=frame_color, width=2 * scale)
        draw.line([(cx, cy + r - s), (cx, cy + r + s)], fill=frame_color, width=2 * scale)
        draw.line([(cx - r - s, cy), (cx - r + s, cy)], fill=frame_color, width=2 * scale)

    elif pos == '數':
        cut = (size - 2 * m) // 3
        poly = [
            (m + cut, m), (size - m - cut, m),
            (size - m, m + cut), (size - m, size - m - cut),
            (size - m - cut, size - m), (m + cut, size - m),
            (m, size - m - cut), (m, m + cut)
        ]
        draw.polygon(poly, outline=frame_color, width=width)

        tick_len = 10 * scale
        draw.line([(cx, m), (cx, m + tick_len)], fill=frame_color, width=width)
        draw.line([(cx, size - m), (cx, size - m - tick_len)], fill=frame_color, width=width)
        draw.line([(m, cy), (m + tick_len, cy)], fill=frame_color, width=width)
        draw.line([(size - m, cy), (size - m - tick_len, cy)], fill=frame_color, width=width)

    elif pos == '代':
        radius = max(10 * scale, (size - 2 * m) // 6)
        draw.rounded_rectangle((m, m, size - m, size - m), radius=radius, outline=frame_color, width=width)
        dot_r = 5 * scale
        draw.ellipse((cx - dot_r, m - dot_r, cx + dot_r, m + dot_r), fill=frame_color)

    elif pos == '虛':
        draw_dashed_ellipse(draw, (m, m, size - m, size - m),
                             outline=frame_color, width=width,
                             dash_len=15 * scale, space_len=10 * scale)

    # 7. 連詞/介詞：左右 85% 高度方括號 ( [ ] )
    elif pos in ['連', '介', '連詞', '介詞']:
        # 1) 計算 85% 的豎線高度與 Y 軸端點
        full_span = size - 2 * m
        line_h = full_span * 0.85
        top_y = cy - line_h / 2
        bot_y = cy + line_h / 2
        
        # 2) 加長橫勾長度，使括號形狀更明顯
        tick = 25 * scale
        
        # 3) 繪製左右垂直線
        draw.line([(m, top_y), (m, bot_y)], fill=frame_color, width=width)            # 左豎線
        draw.line([(size - m, top_y), (size - m, bot_y)], fill=frame_color, width=width) # 右豎線
        
        # 4) 繪製向內延伸的四個短勾
        draw.line([(m, top_y), (m + tick, top_y)], fill=frame_color, width=width)                 # 左上勾
        draw.line([(m, bot_y), (m + tick, bot_y)], fill=frame_color, width=width)                 # 左下勾
        draw.line([(size - m, top_y), (size - m - tick, top_y)], fill=frame_color, width=width)     # 右上勾
        draw.line([(size - m, bot_y), (size - m - tick, bot_y)], fill=frame_color, width=width)     # 右下勾
            
    return hi_res_img.resize((final_size, final_size), Image.Resampling.LANCZOS)

# ==========================================
# 5. 主流程
# ==========================================
def generate_cards():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    output_dir = os.path.join(script_dir, "曼陀羅_png")

    os.makedirs(output_dir, exist_ok=True)
    font_path = os.path.join(script_dir, FONT_FILENAME)
    csv_path = os.path.join(script_dir, CSV_FILENAME)

    if not os.path.exists(font_path):
        print(f"❌ 找不到字體檔 {FONT_FILENAME}")
        return

    if not os.path.exists(csv_path):
        print(f"❌ 找不到資料檔 {CSV_FILENAME}")
        return

    try:
        df = pd.read_csv(csv_path, encoding='big5')
    except Exception:
        df = pd.read_csv(csv_path, encoding='utf-8')

    size = 810
    radius = 432

    y, x = np.ogrid[:size, :size]
    center = size / 2.0
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    norm_dist = np.clip(dist / radius, 0, 1)
    alpha_mask = ((1 - norm_dist)**1.2 * 121).astype(np.uint8)

    standard_font = ImageFont.truetype(font_path, FONT_SIZE)
    success_count = 0

    for idx, row in df.iterrows():
        try:
            char = str(row['字']).strip()
            tone = str(row['聲調']).strip()

            raw_line_idx = row.get('聯句', 1)
            line_idx = int(raw_line_idx) if pd.notna(raw_line_idx) else 1

            raw_rhyme = row.get('韻腳', '')
            is_rhyme = False
            if pd.notna(raw_rhyme):
                try:
                    if int(float(raw_rhyme)) == 1:
                        is_rhyme = True
                except Exception:
                    if str(raw_rhyme).strip() == '1':
                        is_rhyme = True

            raw_pos = row.get('詞性', '')
            pos = '' if pd.isna(raw_pos) else str(raw_pos).strip()
            if pos.lower() in ['nan', 'none', 'null']:
                pos = ''

            color_hex = COLOR_MAP.get(tone, '#CCCCCC')
            r, g, b, _ = hex_to_rgba(color_hex)

            gradient_arr = np.zeros((size, size, 4), dtype=np.uint8)
            gradient_arr[:, :, 0] = r
            gradient_arr[:, :, 1] = g
            gradient_arr[:, :, 2] = b
            gradient_arr[:, :, 3] = alpha_mask
            final_img = Image.fromarray(gradient_arr, mode='RGBA')

            FRAME_COLOR = darken_color(color_hex, factor=0.45, alpha=FRAME_ALPHA)

            if pos:
                frame_layer = generate_supersampled_frame(
                    pos,
                    line_idx,
                    base_margin=BASE_MARGIN,
                    final_size=size,
                    scale=4,
                    frame_color=FRAME_COLOR
                )
                final_img.alpha_composite(frame_layer)

            text_layer = create_stretched_text_layer(
                char, standard_font, (0, 0, 0, 255),
                size, TEXT_STRETCH_Y, is_rhyme=is_rhyme
            )
            final_img.alpha_composite(text_layer)

            pos_label = pos if pos else '無詞性'
            rhyme_label = "_韻" if is_rhyme else ""
            out_name = f"{idx+1:02d}_{char}_{tone}_{pos_label}{rhyme_label}.png"
            final_img.save(os.path.join(output_dir, out_name))
            success_count += 1

        except Exception as e:
            print(f"❌ 第 {idx+1} 列 ({row.get('字', '')}) 錯誤: {e}")

    print(f"\n✅ 完成！成功生成 {success_count} / {len(df)} 張 PNG。")

# ==========================================
# 6. 安全執行入口
# ==========================================
if __name__ == "__main__":
    try:
        generate_cards()
    except Exception as e:
        print("\n❌ 程式執行發生錯誤：\n")
        traceback.print_exc()
        with open("error_log.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("\n📄 詳細錯誤資訊已儲存至 error_log.txt")
    finally:
        input("\n按下 Enter 鍵以關閉視窗...")