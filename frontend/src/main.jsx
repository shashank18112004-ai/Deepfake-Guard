
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App.jsx";
import CandidateInterview from "./CandidateInterview.jsx";
import History from "./History.jsx";

import "./App.css";
import "./CandidateInterview.css";
import "./History.css";

const path = window.location.pathname;

function MainPage() {
  if (path === "/candidate") {
    return <CandidateInterview />;
  }

  if (path === "/history") {
    return <History />;
  }

  return <App />;
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <MainPage />
  </StrictMode>
);
