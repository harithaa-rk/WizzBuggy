// Keyboard shortcuts modal overlay
const SHORTCUTS = [
  { keys: ["Ctrl", "K"],       desc: "Focus chat input" },
  { keys: ["Ctrl", "/"],       desc: "Toggle sidebar" },
  { keys: ["Ctrl", "M"],       desc: "Toggle microphone" },
  { keys: ["Ctrl", "N"],       desc: "New session" },
  { keys: ["Ctrl", "L"],       desc: "Clear conversation" },
  { keys: ["Ctrl", "F"],       desc: "Toggle Search / Chat mode" },
  { keys: ["Escape"],          desc: "Close this panel / clear input" },
  { keys: ["Enter"],           desc: "Send message" },
  { keys: ["Shift", "Enter"],  desc: "New line in message" },
  { keys: ["?"],               desc: "Show keyboard shortcuts" },
];

export default function KeyboardShortcuts({ onClose }) {
  return (
    <div className="kbd-overlay" onClick={onClose}>
      <div className="kbd-card" onClick={(e) => e.stopPropagation()}>
        <div className="kbd-card-title">
          <span>Keyboard Shortcuts</span>
          <button className="kbd-close" onClick={onClose}>✕</button>
        </div>

        {SHORTCUTS.map((s, i) => (
          <div key={i} className="kbd-row">
            <span className="kbd-desc">{s.desc}</span>
            <div className="kbd-keys">
              {s.keys.map((k, j) => (
                <span key={j}>
                  <span className="kbd">{k}</span>
                  {j < s.keys.length - 1 && (
                    <span className="kbd-plus"> + </span>
                  )}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}