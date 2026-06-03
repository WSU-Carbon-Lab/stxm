import { createEnv } from "@t3-oss/env-nextjs";
import { z } from "zod";

export const env = createEnv({
  server: {
    NODE_ENV: z.enum(["development", "test", "production"]),
    STXM_ALLOWED_ROOTS: z.string().optional(),
    STXM_DEFAULT_PARENT_DIR: z.string().optional(),
  },
  client: {},
  runtimeEnv: {
    NODE_ENV: process.env.NODE_ENV,
    STXM_ALLOWED_ROOTS: process.env.STXM_ALLOWED_ROOTS,
    STXM_DEFAULT_PARENT_DIR: process.env.STXM_DEFAULT_PARENT_DIR,
  },
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  emptyStringAsUndefined: true,
});
