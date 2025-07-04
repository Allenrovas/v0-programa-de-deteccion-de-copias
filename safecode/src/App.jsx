import { useState } from "react"
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import NavBar from "./components/NavBar"
import Index from "./pages/Index"
import Info from "./pages/Info"
import Feedback from "./pages/Feedback"
import Analysis from "./pages/Analysis"
import Footer from "./components/Footer"

function App() {
  const [theme, setTheme] = useState("dracula")

  const toggleTheme = () => {
    const newTheme = theme === "autumn" ? "dracula" : "autumn"
    setTheme(newTheme)
    document.documentElement.setAttribute("data-theme", newTheme)
  }

  return (
    <BrowserRouter>
      <NavBar theme={theme} toggleTheme={toggleTheme} />
      <Routes>
        <Route path="/" element={<Index theme={theme} toggleTheme={toggleTheme} />} />
        <Route path="/info" element={<Info theme={theme} toggleTheme={toggleTheme} />} />
        <Route path="/feedback" element={<Feedback theme={theme} toggleTheme={toggleTheme} />} />
        <Route path="/analysis" element={<Analysis theme={theme} toggleTheme={toggleTheme} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <Footer theme={theme} />
    </BrowserRouter>
  )
}

export default App
