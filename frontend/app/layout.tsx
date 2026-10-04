import type { Metadata } from "next";
import "./globals.css";
import { AppShell } from "@/components/app-shell";

export const metadata: Metadata = {
  title: "RAG Precision Engine · ADROSONIC",
  description: "Large-scale hybrid retrieval intelligence for grounded RAG systems.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" data-scroll-behavior="smooth">
      <body className="min-h-screen"><AppShell>{children}</AppShell></body>
    </html>
  );
}
