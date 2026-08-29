import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pre-Impact Fall Anticipation",
  description:
    "Predict a fall before impact from skeleton data, and name the joints that signalled it — " +
    "with a deletion/insertion test that checks whether that evidence is faithful. Research prototype.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
