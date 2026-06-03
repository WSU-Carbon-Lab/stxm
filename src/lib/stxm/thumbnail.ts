import "server-only";

import sharp from "sharp";

function percentile(values: number[], p: number): number {
  const finite = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (finite.length === 0) {
    return Number.NaN;
  }
  const idx = (p / 100) * (finite.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) {
    return finite[lo] ?? Number.NaN;
  }
  const weight = idx - lo;
  return (finite[lo] ?? 0) * (1 - weight) + (finite[hi] ?? 0) * weight;
}

/**
 * Encode a downsampled grayscale preview of a 2D scan as base64 PNG.
 */
export async function thumbnailPngBase64(
  image: number[][],
  maxSize = 128,
): Promise<string> {
  if (image.length === 0 || (image[0]?.length ?? 0) === 0) {
    return "";
  }
  const height = image.length;
  const width = image[0]?.length ?? 0;
  const scale = Math.max(height, width) / maxSize;
  const step = scale > 1 ? Math.ceil(scale) : 1;
  const preview: number[][] = [];
  for (let row = 0; row < height; row += step) {
    const previewRow: number[] = [];
    for (let col = 0; col < width; col += step) {
      previewRow.push(image[row]?.[col] ?? 0);
    }
    preview.push(previewRow);
  }
  const flatPreview = preview.flat();
  let vmin = percentile(flatPreview, 2);
  let vmax = percentile(flatPreview, 98);
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmin >= vmax) {
    vmin = Math.min(...flatPreview.filter(Number.isFinite));
    vmax = Math.max(...flatPreview.filter(Number.isFinite));
    if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmin >= vmax) {
      vmin = 0;
      vmax = 1;
    }
  }
  const pHeight = preview.length;
  const pWidth = preview[0]?.length ?? 0;
  const buffer = Buffer.alloc(pWidth * pHeight);
  for (let row = 0; row < pHeight; row += 1) {
    for (let col = 0; col < pWidth; col += 1) {
      const value = preview[row]?.[col] ?? 0;
      const normalized = (value - vmin) / (vmax - vmin);
      const clamped = Math.max(0, Math.min(1, normalized));
      buffer[row * pWidth + col] = Math.round(clamped * 255);
    }
  }
  const png = await sharp(buffer, {
    raw: { width: pWidth, height: pHeight, channels: 1 },
  })
    .png()
    .toBuffer();
  return png.toString("base64");
}
