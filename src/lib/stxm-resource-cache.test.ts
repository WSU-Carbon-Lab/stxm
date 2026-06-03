import { describe, expect, test } from "bun:test";

import {
  StxmResourceCache,
  catalogCacheKey,
  scanCacheKey,
} from "@/lib/stxm-resource-cache";

describe("catalogCacheKey", () => {
  test("trims experiment directory paths", () => {
    expect(catalogCacheKey("/data/exp/ ")).toBe("/data/exp/");
  });
});

describe("scanCacheKey", () => {
  test("trims hdr paths", () => {
    expect(scanCacheKey("/data/exp/scan.hdr ")).toBe("/data/exp/scan.hdr");
  });
});

describe("StxmResourceCache", () => {
  test("stores and retrieves values by key", () => {
    const cache = new StxmResourceCache<string[]>();
    cache.set("a", ["one"]);
    expect(cache.get("a")).toEqual(["one"]);
    expect(cache.get("missing")).toBeUndefined();
  });

  test("delete removes an entry", () => {
    const cache = new StxmResourceCache<number>();
    cache.set("x", 1);
    cache.delete("x");
    expect(cache.get("x")).toBeUndefined();
  });
});
