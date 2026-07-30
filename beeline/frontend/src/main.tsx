import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

// No StrictMode: it double-mounts effects in dev, which makes the imperative
// YouTube IFrame player boot twice and occasionally lose its handle.
createRoot(document.getElementById("root")!).render(<App />);
