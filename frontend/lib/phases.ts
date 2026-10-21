/**
 * The four phases and their colours, shared by every component that draws them.
 *
 * Okabe-Ito palette: the four hues stay distinguishable under the common forms of
 * colour-vision deficiency. Every use must pair the colour with the text label -
 * colour alone is never the only carrier of which phase is which.
 */
export const PHASES = [
  { name: "normal", colour: "#8A6E5C", blurb: "ordinary activity" },
  { name: "imminent", colour: "#C98A00", blurb: "balance about to be lost" },
  { name: "falling", colour: "#C2540A", blurb: "on the way to the floor" },
  { name: "grounded", colour: "#5A0E24", blurb: "already down" },
] as const;

export type PhaseName = (typeof PHASES)[number]["name"];
