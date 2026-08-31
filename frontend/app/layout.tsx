import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FBimpact · Pre-Impact Fall Anticipation",
  description:
    "Predict a fall before impact from privacy-preserving skeleton data, and name the joints " +
    "whose instability signalled it — with a deletion/insertion test that checks whether that " +
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
  themeColor: "#05070c",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
