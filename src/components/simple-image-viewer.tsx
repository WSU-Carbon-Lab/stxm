"use client";

import { useEffect, useRef } from "react";

import {
  lineScanImageDisplayScale,
  lineScanPixelGray,
  type PlotScaleMode,
} from "@/lib/stxm/plot-scale";

type SimpleImageViewerProps = {
  image: number[][];
  imageMin: number;
  imageMax: number;
  paxisName: string;
  qaxisName: string;
  title: string;
  scanType: string;
  shape: number[];
  imageScaleMode?: PlotScaleMode;
};

export function SimpleImageViewer({
  image,
  imageMin,
  imageMax,
  paxisName,
  qaxisName,
  title,
  scanType,
  shape,
  imageScaleMode = "linear",
}: SimpleImageViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || image.length === 0) {
      return;
    }
    const rows = image.length;
    const cols = image[0]?.length ?? 0;
    if (cols === 0) {
      return;
    }
    canvas.width = cols;
    canvas.height = rows;
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    const displayScale = lineScanImageDisplayScale(image, imageMin, imageMax, imageScaleMode);
    const pixels = context.createImageData(cols, rows);
    for (let row = 0; row < rows; row += 1) {
      const rowValues = image[row] ?? [];
      for (let col = 0; col < cols; col += 1) {
        const value = rowValues[col] ?? imageMin;
        const gray = lineScanPixelGray(value, displayScale);
        const offset = (row * cols + col) * 4;
        pixels.data[offset] = gray;
        pixels.data[offset + 1] = gray;
        pixels.data[offset + 2] = gray;
        pixels.data[offset + 3] = 255;
      }
    }
    context.putImageData(pixels, 0, 0);
  }, [image, imageMax, imageMin, imageScaleMode]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div>
        <h3 className="font-mono text-sm font-medium text-zinc-900">{title}</h3>
        <p className="text-sm text-zinc-600">{scanType}</p>
        <p className="text-xs text-zinc-500">
          {shape[0]} x {shape[1]} ({qaxisName} vertical x {paxisName} horizontal)
        </p>
      </div>
      <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto rounded-lg border border-zinc-200 bg-zinc-950 p-3">
        <canvas
          ref={canvasRef}
          className="max-h-full max-w-full object-contain"
          style={{ imageRendering: "pixelated" }}
        />
      </div>
    </div>
  );
}
