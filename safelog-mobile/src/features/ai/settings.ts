// Expo replaces these direct property accesses when it builds the app.
// These are server addresses, never API keys or model weights.
export const fixedCpuSettings = {
  generalBaseUrl: process.env.EXPO_PUBLIC_SAFELOG_AI_BASE_URL ?? 'http://127.0.0.1:8780/general',
  pcbBaseUrl: process.env.EXPO_PUBLIC_SAFELOG_PCB_BASE_URL ?? 'http://127.0.0.1:8780/pcb',
};
