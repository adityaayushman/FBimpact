"use client";

/**
 * Video -> skeleton, entirely in the browser.
 *
 * The file is read into an object URL, decoded frame by frame, posed with
 * MoveNet and discarded. Nothing is uploaded: the API has no endpoint that
 * accepts an image, so the privacy property is structural rather than a promise
 * about what the server does with what it receives.
 */

import { useRef, useState } from "react";
import type { Keypoints } from "@/lib/api";
import { extractFromVideo } from "@/lib/pose";

interface Props {
  maxFrames: number;
  disabled?: boolean;
  onExtracted: (result: { keypoints: Keypoints; fps: number; name: string }) => void;
  onError: (message: string) => void;
}

export default function VideoUpload({ maxFrames, disabled, onExtracted, onError }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [busy, setBusy] = useState(false);

  const handleFile = async (file: File) => {
    const video = videoRef.current;
    if (!video) return;

    setBusy(true);
    setProgress({ done: 0, total: 1 });
    const url = URL.createObjectURL(file);

    try {
      video.src = url;
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve();
        video.onerror = () => reject(new Error("could not decode that video"));
      });

      const result = await extractFromVideo(video, {
        maxFrames,
        onProgress: (done, total) => setProgress({ done, total }),
      });
      onExtracted({ keypoints: result.keypoints, fps: result.fps, name: file.name });
    } catch (error) {
      onError(error instanceof Error ? error.message : String(error));
    } finally {
      URL.revokeObjectURL(url);
      video.removeAttribute("src");
      setBusy(false);
      setProgress(null);
    }
  };

  const pct = progress && progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;

  return (
    <div>
      <label className="file-drop">
        <input
          type="file"
          accept="video/*"
          disabled={disabled || busy}
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) void handleFile(file);
          }}
        />
        {busy ? `Estimating poses… ${pct}%` : "Choose a video file"}
      </label>

      <p className="hint" style={{ marginTop: 8, marginBottom: 0 }}>
        Posed in your browser with MoveNet; the video never leaves this device. Frame rate is
        assumed to be 30 fps — lead time is in seconds, so a different source rate scales it.
      </p>

      {/* Never displayed: it exists only as a decode target. */}
      <video ref={videoRef} muted playsInline preload="auto" style={{ display: "none" }} />
    </div>
  );
}
