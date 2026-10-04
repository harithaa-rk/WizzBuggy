import { useState, useRef, useCallback } from "react";

export function useVoice(onResult, apiUrl = "http://127.0.0.1:8000") {
  const [recording, setRecording] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [supported]               = useState(
    () => typeof window !== "undefined" && !!navigator.mediaDevices?.getUserMedia
  );

  const mediaRecRef  = useRef(null);
  const streamRef    = useRef(null);
  const chunksRef    = useRef([]);
  const onResultRef  = useRef(onResult);
  onResultRef.current = onResult;

  const stop = useCallback(async () => {
    if (mediaRecRef.current && mediaRecRef.current.state !== "inactive") {
      mediaRecRef.current.stop();   // triggers onstop → sends audio
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  }, []);

  const start = useCallback(async () => {
    if (!supported) return;
    chunksRef.current = [];
    setStatusMsg("Requesting mic…");

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      console.log("[Voice] Mic opened:", stream.getAudioTracks().map((t) => t.label));
    } catch (err) {
      setStatusMsg("");
      if (err.name === "NotAllowedError") alert("Microphone permission denied.");
      else if (err.name === "NotFoundError") alert("No microphone found.");
      else alert("Mic error: " + err.message);
      return;
    }

    // Pick best supported format
    const mimeType = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg",
    ].find((m) => MediaRecorder.isTypeSupported(m)) || "";

    console.log("[Voice] Using MIME type:", mimeType || "browser default");

    const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    mediaRecRef.current = rec;

    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        chunksRef.current.push(e.data);
        console.log("[Voice] Chunk received, size:", e.data.size);
      }
    };

    rec.onstart = () => {
      setRecording(true);
      setStatusMsg("Recording — speak now");
      console.log("[Voice] MediaRecorder started");
    };

    rec.onstop = async () => {
      setRecording(false);
      setStatusMsg("Transcribing…");
      console.log("[Voice] Stopped, chunks:", chunksRef.current.length);

      if (chunksRef.current.length === 0) {
        setStatusMsg("No audio captured");
        setTimeout(() => setStatusMsg(""), 2000);
        return;
      }

      const blob = new Blob(chunksRef.current, {
        type: mimeType || "audio/webm",
      });
      console.log("[Voice] Blob size:", blob.size, "type:", blob.type);

      // Send to your Whisper backend
      const form = new FormData();
      form.append("file", blob, "recording.webm");

      try {
        const res = await fetch(`${apiUrl}/transcribe`, {
          method: "POST",
          body: form,
        });
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        const transcript = data.text?.trim();
        console.log("[Voice] Transcript:", transcript);

        if (transcript) {
          setStatusMsg("");
          onResultRef.current(transcript);
        } else {
          setStatusMsg("Could not understand — try again");
          setTimeout(() => setStatusMsg(""), 2500);
        }
      } catch (err) {
        console.error("[Voice] Transcription failed:", err);
        setStatusMsg("Transcription failed: " + err.message);
        setTimeout(() => setStatusMsg(""), 3000);
      }
    };

    rec.onerror = (e) => {
      console.error("[Voice] MediaRecorder error:", e);
      setRecording(false);
      setStatusMsg("Recording error");
      setTimeout(() => setStatusMsg(""), 2000);
    };

    rec.start(250); // collect chunks every 250ms
  }, [supported, apiUrl]);

  const toggle = useCallback(() => {
    recording ? stop() : start();
  }, [recording, start, stop]);

  return { recording, supported, toggle, statusMsg };
}