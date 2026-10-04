function baseName(p) {
  if (!p) return "Unknown";
  return p.split(/[/\\]/).pop();
}

function formatTimestamp() {
  return new Date().toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

export function buildMarkdown(history, sessionName) {
  const lines = [];
  lines.push("# NovaRAG Chat Export");
  lines.push("**Session:** " + (sessionName || "Unnamed"));
  lines.push("**Exported:** " + formatTimestamp());
  lines.push("**Messages:** " + history.length);
  lines.push("");
  lines.push("---");
  lines.push("");

  for (const item of history) {
    if (item.role === "user") {
      lines.push("## You");
      lines.push("");
      lines.push(item.message || "");
      lines.push("");
    } else {
      lines.push("## NovaRAG");
      lines.push("");
      lines.push(item.message || "");
      lines.push("");

      if (item.confidence != null) {
        lines.push("> **Confidence:** " + item.confidence + "%  |  **Chunks:** " + (item.chunks_used ?? "?"));
        lines.push("");
      }

      if (item.citations && item.citations.length > 0) {
        const seen   = new Set();
        const unique = item.citations.filter((c) => {
          const key = c.path + "|" + c.page;
          if (seen.has(key)) return false;
          seen.add(key); return true;
        });
        lines.push("**Sources:**");
        lines.push("");
        for (const c of unique) {
          const score = Math.round((Number(c.score) || 0) * 100);
          const page  = c.page ? " (p." + c.page + ")" : "";
          lines.push("- `" + baseName(c.path) + "`" + page + " — " + score + "% relevance");
          if (c.snippet) {
            const preview = c.snippet.slice(0, 120).replace(/\n/g, " ");
            lines.push("  > " + preview + (c.snippet.length > 120 ? "…" : ""));
          }
        }
        lines.push("");
      }
      lines.push("---");
      lines.push("");
    }
  }
  return lines.join("\n");
}

function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function safeName(sessionName) {
  return (sessionName || "chat").replace(/[^a-z0-9]/gi, "_").toLowerCase();
}

export function exportAsMarkdown(history, sessionName) {
  downloadFile(
    buildMarkdown(history, sessionName),
    "novarag_" + safeName(sessionName) + "_" + Date.now() + ".md",
    "text/markdown"
  );
}

export function exportAsText(history, sessionName) {
  const text = buildMarkdown(history, sessionName)
    .replace(/^#{1,3} /gm, "")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/`(.*?)`/g, "$1")
    .replace(/^> /gm, "  ");
  downloadFile(
    text,
    "novarag_" + safeName(sessionName) + "_" + Date.now() + ".txt",
    "text/plain"
  );
}

export function exportAsJSON(history, sessionName) {
  downloadFile(
    JSON.stringify({ session: sessionName || "Unnamed", exported: new Date().toISOString(), messages: history.length, history }, null, 2),
    "novarag_" + safeName(sessionName) + "_" + Date.now() + ".json",
    "application/json"
  );
}