import React, { useState } from 'react';
import Presentation from '../components/Presentation';

function Index() {
  const [theme, setTheme] = useState('autumn');

  const toggleTheme = () => {
    const newTheme = theme === 'autumn' ? 'dracula' : 'autumn';
    setTheme(newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

  return (
    <div className="App">
      <Presentation />
    </div>
  );
}

export default Index;