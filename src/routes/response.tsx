import { createFileRoute } from "@tanstack/react-router";
import DefenderResponse from "@/pages/DefenderResponse";

export const Route = createFileRoute("/response")({
  head: () => ({
    meta: [
      { title: "Defender Response & Verification — NETWORLD" },
      {
        name: "description",
        content:
          "Operator-approved controlled lab containment and closed-loop empirical verification with the PyTorch LSTM World Model.",
      },
      { property: "og:title", content: "Defender Response & Verification — NETWORLD" },
      {
        property: "og:description",
        content:
          "Operator-approved controlled lab containment and closed-loop empirical verification with the PyTorch LSTM World Model.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: DefenderResponse,
});
