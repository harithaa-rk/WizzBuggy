import { useEffect, useRef } from "react";

/**
 * Arc gauge — sweeps 240° (from -210° to +30° visually).
 * confidence: 0-100
 */
export default function ConfidenceGauge({ confidence, chunksUsed }) {
  const fillRef = useRef(null);

  // Arc math: r=18, circumference of full 240° arc
  const r           = 18;
  const arcDeg      = 240;
  const arcRad      = (arcDeg * Math.PI) / 180;
  const circumference = r * arcRad;          // ~75.4

  // colour: red → amber → teal based on value
  const colour =
    confidence >= 70 ? "var(--teal)"
    : confidence >= 40 ? "var(--amber)"
    : "var(--rose)";

  const offset = circumference - (confidence / 100) * circumference;

  useEffect(() => {
    if (!fillRef.current) return;
    // start fully hidden, then animate to value
    fillRef.current.style.strokeDashoffset = String(circumference);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (fillRef.current)
          fillRef.current.style.strokeDashoffset = String(offset);
      });
    });
  }, [confidence, circumference, offset]);

  return (
    <div className="gauge-wrap">
      <svg
        className="gauge-svg"
        width="54"
        height="54"
        viewBox="0 0 54 54"
      >
        {/* track arc */}
        <circle
          className="gauge-track"
          cx="27" cy="27" r={r}
          strokeDasharray={`${circumference} ${2 * Math.PI * r}`}
          strokeDashoffset={0}
          style={{ transform: "rotate(-210deg)", transformOrigin: "center" }}
        />
        {/* fill arc */}
        <circle
          ref={fillRef}
          className="gauge-fill"
          cx="27" cy="27" r={r}
          stroke={colour}
          strokeDasharray={`${circumference} ${2 * Math.PI * r}`}
          strokeDashoffset={circumference}
          style={{
            filter: `drop-shadow(0 0 4px ${colour})`,
            transform: "rotate(-210deg)",
            transformOrigin: "center",
            transition: "stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1), stroke 0.4s ease",
          }}
        />
        {/* centre label */}
        <text className="gauge-label" x="27" y="25" style={{ fill: colour }}>
          {confidence}
        </text>
        <text className="gauge-sub" x="27" y="34">
          %
        </text>
      </svg>

      <div className="gauge-meta">
        <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>
          Confidence
        </span>
        {chunksUsed != null && (
          <span className="gauge-chunks">{chunksUsed} chunks used</span>
        )}
        <span className="gauge-chunks" style={{ color: colour }}>
          {confidence >= 70 ? "High" : confidence >= 40 ? "Medium" : "Low"}
        </span>
      </div>
    </div>
  );
}