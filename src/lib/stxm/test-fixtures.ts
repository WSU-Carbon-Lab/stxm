import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export function writeLineScanFixture(
  experimentDir: string,
  stem: string,
  scanType = "NEXAFS Line Scan",
): { hdrPath: string; ximPath: string } {
  const nEnergy = 5;
  const nSpatial = 8;
  const energy = Array.from({ length: nEnergy }, (_, idx) => 280 + idx);
  const spatial = Array.from({ length: nSpatial }, (_, idx) => idx);
  const hdrPath = path.join(experimentDir, `${stem}.hdr`);
  const ximPath = path.join(experimentDir, `${stem}_a.xim`);
  fs.writeFileSync(
    hdrPath,
    [
      `Type = "${scanType}"`,
      `PAxis = { Name = "Energy (eV)" Points = ( ${nEnergy} , ${energy.map((value) => value.toFixed(1)).join(" ")} ) }`,
      `QAxis = { Name = "Sample" Points = ( ${nSpatial} , ${spatial.map((value) => value.toFixed(1)).join(" ")} ) }`,
    ].join("\n"),
  );
  const image: number[][] = [];
  let counter = 1;
  for (let row = 0; row < nSpatial; row += 1) {
    const rowValues: number[] = [];
    for (let col = 0; col < nEnergy; col += 1) {
      rowValues.push(counter);
      counter += 1;
    }
    image.push(rowValues);
  }
  fs.writeFileSync(
    ximPath,
    image.map((row) => row.map((value) => value.toFixed(6)).join(" ")).join("\n"),
  );
  return { hdrPath, ximPath };
}

export function writeImageScanFixture(
  experimentDir: string,
  stem: string,
  energy = 284.2,
): { hdrPath: string; ximPath: string } {
  const nX = 10;
  const nY = 10;
  const xAxis = Array.from({ length: nX }, (_, idx) => idx);
  const yAxis = Array.from({ length: nY }, (_, idx) => idx);
  const hdrPath = path.join(experimentDir, `${stem}.hdr`);
  const ximPath = path.join(experimentDir, `${stem}_a.xim`);
  fs.writeFileSync(
    hdrPath,
    [
      'Type = "Image Scan"',
      `Energy = ${energy.toFixed(1)}`,
      `PAxis = { Name = "X (um)" Points = ( ${nX} , ${xAxis.map((value) => value.toFixed(1)).join(" ")} ) }`,
      `QAxis = { Name = "Y (um)" Points = ( ${nY} , ${yAxis.map((value) => value.toFixed(1)).join(" ")} ) }`,
    ].join("\n"),
  );
  const image: number[][] = [];
  let counter = 1;
  for (let row = 0; row < nY; row += 1) {
    const rowValues: number[] = [];
    for (let col = 0; col < nX; col += 1) {
      rowValues.push(counter);
      counter += 1;
    }
    image.push(rowValues);
  }
  fs.writeFileSync(
    ximPath,
    image.map((row) => row.map((value) => value.toFixed(6)).join(" ")).join("\n"),
  );
  return { hdrPath, ximPath };
}

export function withTempExperiment(run: (root: string, experimentDir: string) => void): void {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "stxm-ts-test-"));
  try {
    const experimentDir = path.join(root, "2024-01(Jan)");
    fs.mkdirSync(experimentDir, { recursive: true });
    run(root, experimentDir);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}
