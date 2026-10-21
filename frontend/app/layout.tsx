import type { Metadata, Viewport } from "next";
import { Fraunces, Inter } from "next/font/google";
import "./globals.css";

// Inter for the interface, Fraunces for page and section titles. Loaded through
// next/font, so they are self-hosted at build time and never block rendering.
const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });
const display = Fraunces({ subsets: ["latin"], variable: "--font-display", display: "swap" });

export const metadata: Metadata = {
  title: "FBimpact · Pre-Impact Fall Anticipation",
  description:
    "Predict a fall before impact from privacy-preserving skeleton data, and name the joints " +
    "whose instability signalled it, with a deletion/insertion test that checks whether that " +
    "evidence is faithful. Research prototype; not a medical device.",
  applicationName: "FBimpact",
  authors: [{ name: "Aditya Sahoo" }],
  keywords: [
    "fall prediction", "pre-impact fall detection", "skeleton action recognition",
    "ST-GCN", "explainable AI", "faithfulness", "elderly care", "privacy-preserving vision",
  ],
  openGraph: {
    title: "Pre-Impact Fall Anticipation with Grounded Skeletal Evidence",
    description:
      "Vision-based, privacy-preserving fall anticipation with joint-level evidence that is " +
      "scored for faithfulness rather than asserted.",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#5A0E24",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${display.variable}`}>
      <body>{children}</body>
    </html>
  );
}
