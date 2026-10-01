export function PrivacyNote({ cloudOnly = false }: { cloudOnly?: boolean }) {
  return (
    <p className="privacy-note">
      {cloudOnly
        ? "Transcription runs on OpenRouter's cloud: audio is uploaded for processing and billed per second. "
        : "Processing is performed locally on this Mac. Your lecture is not uploaded anywhere. "}
      No telemetry, no analytics, no remote services beyond the transcription backend you choose.
    </p>
  );
}
