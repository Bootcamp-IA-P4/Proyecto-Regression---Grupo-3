import React from 'react';
import './App.css';
import PredictForm from './PredictForm';  // Importa tu componente

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Calculadora de Precios - Airbnb Madrid</h1>
        <PredictForm />  {/* Usa tu componente aquí */}
      </header>
    </div>
  );
}

export default App;