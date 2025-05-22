import React from 'react';
import { Link } from 'react-router-dom';

function NavBar({ theme, toggleTheme }) {
  return (
    <nav className="navbar bg-base-100 shadow-lg">
      <div className="flex-1">
        <Link to="/" className="btn btn-ghost normal-case text-xl">Safe Code</Link>
      </div>
      <div className="flex-none">
        <ul className="menu menu-horizontal px-1">
          <li><Link to="/info">Información</Link></li>
          <li><Link to="/feedback">Retroalimentación</Link></li>
          <li><Link to="/analysis">Análisis</Link></li>
        </ul>
        <label className="swap swap-rotate">
          <input type="checkbox" onChange={toggleTheme} checked={theme === 'dracula'} />
          <svg className="swap-on fill-current w-10 h-10" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M5 15a7 7 0 0 0 14 0H5z" />
          </svg>
          <svg className="swap-off fill-current w-10 h-10" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M12 3v1m0 16v1m8.66-8.66h-1M4.34 12h-1m15.07-5.07l-.71.71M6.34 17.66l-.71.71m12.02-12.02l-.71.71M6.34 6.34l-.71.71" />
          </svg>
        </label>
      </div>
    </nav>
  );
}

export default NavBar;