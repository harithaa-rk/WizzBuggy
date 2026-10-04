export const Path = {
  basename: (p) => {
    if (!p) return "Unknown";
    return p.split(/[/\\]/).pop();
  }
};
