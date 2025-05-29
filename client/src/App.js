import logo from './logo.svg';
import './App.css';
import Navigation from './Navigation';
import Topbar from './components/Topbar';
import { BrowserRouter } from 'react-router-dom';

function App() {
  return (
    <BrowserRouter>
      <div className="App">
        <Topbar />
        <Navigation/>
      </div>
    </BrowserRouter>
  );
}

export default App;
