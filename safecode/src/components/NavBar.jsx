import { useState } from "react"
import { Link } from "react-router-dom"
import { Sun, Moon, Heart, Menu, X } from "lucide-react"

function NavBar({ theme, toggleTheme }) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const isDark = theme === "dracula"

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen)
  }

  return (
    <nav
      className={`backdrop-blur-md shadow-lg border-b sticky top-0 z-50 transition-colors duration-300 ${
        isDark ? "bg-gray-900/95 border-gray-700" : "bg-white/95 border-gray-100"
      }`}
    >
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo and Brand */}
          <div className="flex items-center space-x-4">
            <Link
              to="/"
              className="flex items-center space-x-2 text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent hover:from-blue-700 hover:to-purple-700 transition-all duration-300"
            >
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
                <Heart className="w-6 h-6 text-white" />
              </div>
              Safe Code
            </Link>

            {/* Non-profit badge */}
            <span
              className={`hidden sm:flex text-xs font-medium px-2 py-1 rounded-full transition-colors duration-300 ${
                isDark ? "bg-red-900/50 text-red-300 border border-red-700" : "bg-red-100 text-red-800"
              }`}
            >
              Sin Fines de Lucro
            </span>
          </div>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-1">
            <Link
              to="/info"
              className={`px-4 py-2 rounded-lg transition-all duration-300 font-medium ${
                isDark
                  ? "text-gray-300 hover:text-blue-400 hover:bg-gray-800"
                  : "text-gray-700 hover:text-blue-600 hover:bg-blue-50"
              }`}
            >
              Información
            </Link>
            <Link
              to="/feedback"
              className={`px-4 py-2 rounded-lg transition-all duration-300 font-medium ${
                isDark
                  ? "text-gray-300 hover:text-blue-400 hover:bg-gray-800"
                  : "text-gray-700 hover:text-blue-600 hover:bg-blue-50"
              }`}
            >
              Retroalimentación
            </Link>
            <Link
              to="/analysis"
              className={`px-4 py-2 rounded-lg transition-all duration-300 font-medium ${
                isDark
                  ? "text-gray-300 hover:text-blue-400 hover:bg-gray-800"
                  : "text-gray-700 hover:text-blue-600 hover:bg-blue-50"
              }`}
            >
              Análisis
            </Link>
          </div>

          {/* Theme Toggle and Mobile Menu */}
          <div className="flex items-center space-x-4">
            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg transition-all duration-300 transform hover:scale-110 ${
                isDark ? "bg-gray-800 hover:bg-gray-700" : "bg-gray-100 hover:bg-gray-200"
              }`}
              aria-label="Toggle theme"
            >
              {isDark ? <Sun className="w-5 h-5 text-yellow-500" /> : <Moon className="w-5 h-5 text-gray-600" />}
            </button>

            {/* Mobile Menu Button */}
            <div className="md:hidden">
              <button
                onClick={toggleMobileMenu}
                className={`p-2 rounded-lg transition-all duration-300 ${
                  isDark ? "hover:bg-gray-800 text-gray-300" : "hover:bg-gray-100 text-gray-600"
                }`}
                aria-label="Toggle mobile menu"
              >
                {isMobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>

        {/* Mobile Navigation Menu */}
        <div
          className={`md:hidden border-t transition-all duration-300 overflow-hidden ${
            isMobileMenuOpen ? "max-h-64 py-4" : "max-h-0 py-0"
          } ${isDark ? "border-gray-700" : "border-gray-100"}`}
        >
          <div className="flex flex-col space-y-2">
            <Link
              to="/info"
              onClick={() => setIsMobileMenuOpen(false)}
              className={`px-4 py-2 rounded-lg transition-all duration-300 font-medium ${
                isDark
                  ? "text-gray-300 hover:text-blue-400 hover:bg-gray-800"
                  : "text-gray-700 hover:text-blue-600 hover:bg-blue-50"
              }`}
            >
              Información
            </Link>
            <Link
              to="/feedback"
              onClick={() => setIsMobileMenuOpen(false)}
              className={`px-4 py-2 rounded-lg transition-all duration-300 font-medium ${
                isDark
                  ? "text-gray-300 hover:text-blue-400 hover:bg-gray-800"
                  : "text-gray-700 hover:text-blue-600 hover:bg-blue-50"
              }`}
            >
              Retroalimentación
            </Link>
            <Link
              to="/analysis"
              onClick={() => setIsMobileMenuOpen(false)}
              className={`px-4 py-2 rounded-lg transition-all duration-300 font-medium ${
                isDark
                  ? "text-gray-300 hover:text-blue-400 hover:bg-gray-800"
                  : "text-gray-700 hover:text-blue-600 hover:bg-blue-50"
              }`}
            >
              Análisis
            </Link>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default NavBar
