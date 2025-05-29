import Presentation from "../components/Presentation"

function Index({ theme, toggleTheme }) {
  return (
    <div className="App">
      <Presentation theme={theme} toggleTheme={toggleTheme} />
    </div>
  )
}

export default Index
