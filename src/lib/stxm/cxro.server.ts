import "server-only";

/**
 * CXRO bare-atom mass absorption mu/rho (cm^2/g) on an experiment energy grid.
 *
 * Tabulated values are computed in Python via `periodictable.xsf.index_of_refraction`, which
 * uses CXRO optical constants (beta = -Im n at rho = 1 g/cm^3). xray-atlas instead fetches
 * Henke/CXRO f1,f2 per element and forms mu/rho = 2*r_e*lambda*N_A*f2/M; both are standard
 * bare-atom references before step-edge fitting.
 *
 * @see https://github.com/WSU-Carbon-Lab/xray-atlas/tree/main/src/features/process-nexafs
 */

import { runStxmBridge } from "@/lib/python-bridge.server";

const muRhoCache = new Map<string, Promise<number[]>>();

function cacheKey(formula: string, energyEv: number[]): string {
  const energyKey = energyEv.map((value) => value.toFixed(6)).join(",");
  return `${formula.trim()}::${energyKey}`;
}

/**
 * Load stoichiometry-weighted bare-atom mass absorption mu/rho (cm^2/g) at each energy in eV.
 *
 * @param formula - Chemical formula string (e.g. `C`, `H2O`, `C8H8`).
 * @param energyEv - Photon energies in eV; same grid as the reduced spectrum.
 * @returns Mass absorption coefficient mu/rho in cm^2/g per energy point.
 * @throws Error when the formula is empty or the Python bridge rejects the request.
 */
export async function massAbsorptionCm2PerG(
  formula: string,
  energyEv: number[],
): Promise<number[]> {
  const cleaned = formula.trim();
  if (!cleaned) {
    throw new Error("Chemical formula is required for CXRO mass absorption");
  }
  if (energyEv.length === 0) {
    return [];
  }
  const key = cacheKey(cleaned, energyEv);
  const existing = muRhoCache.get(key);
  if (existing) {
    return existing;
  }
  const pending = runStxmBridge<{ ok: true; mu_rho_cm2_per_g: number[] }>("mass-absorption", [
    "--formula",
    cleaned,
    "--energy-json",
    JSON.stringify(energyEv),
  ]).then((payload) => {
    if (!Array.isArray(payload.mu_rho_cm2_per_g)) {
      throw new Error("Invalid mass-absorption response from stxm-bridge");
    }
    return payload.mu_rho_cm2_per_g;
  });
  muRhoCache.set(
    key,
    pending.catch((error) => {
      muRhoCache.delete(key);
      throw error;
    }),
  );
  return pending;
}
