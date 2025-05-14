// Main App Component
const App = () => {
  // State for app data
  const [strategies, setStrategies] = React.useState([]);
  const [symbols, setSymbols] = React.useState([]);
  const [backtests, setBacktests] = React.useState([]);
  const [simulations, setSimulations] = React.useState([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(null);
  const [activeTab, setActiveTab] = React.useState('dashboard');
  
  // Fetch data on component mount
  React.useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('Fetching data...');
        setLoading(true);
        
        // Fetch strategies
        const strategiesResponse = await fetch('/api/strategies');
        const strategiesData = await strategiesResponse.json();
        console.log('Strategies:', strategiesData);
        setStrategies(strategiesData);
        
        // Fetch symbols
        const symbolsResponse = await fetch('/api/symbols');
        const symbolsData = await symbolsResponse.json();
        console.log('Symbols:', symbolsData);
        setSymbols(symbolsData);
        
        // Fetch backtests
        const backtestsResponse = await fetch('/api/backtests');
        const backtestsData = await backtestsResponse.json();
        console.log('Backtests:', backtestsData);
        setBacktests(backtestsData);
        
        // Fetch simulations
        try {
          const simulationsResponse = await fetch('/api/simulations');
          if (simulationsResponse.ok) {
            const simulationsData = await simulationsResponse.json();
            console.log('Simulations:', simulationsData);
            setSimulations(simulationsData);
          }
        } catch (e) {
          console.warn('Error fetching simulations:', e);
          // Continue even if simulations fail
        }
        
        setLoading(false);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load data. Please try again later.');
        setLoading(false);
      }
    };
    
    fetchData();
  }, []);
  
  // Simple Navbar component
  const SimpleNavbar = () => (
    <nav className="navbar navbar-expand-lg navbar-dark bg-primary">
      <div className="container-fluid">
        <a className="navbar-brand" href="#">KITE Trading System</a>
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav">
            <li className="nav-item">
              <a className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`} 
                 href="#" onClick={() => setActiveTab('dashboard')}>Dashboard</a>
            </li>
            <li className="nav-item">
              <a className={`nav-link ${activeTab === 'strategies' ? 'active' : ''}`} 
                 href="#" onClick={() => setActiveTab('strategies')}>Strategies</a>
            </li>
            <li className="nav-item">
              <a className={`nav-link ${activeTab === 'backtest' ? 'active' : ''}`} 
                 href="#" onClick={() => setActiveTab('backtest')}>Backtest</a>
            </li>
            <li className="nav-item">
              <a className={`nav-link ${activeTab === 'livesimulation' ? 'active' : ''}`} 
                 href="#" onClick={() => setActiveTab('livesimulation')}>Live Simulation</a>
            </li>
            <li className="nav-item">
              <a className={`nav-link ${activeTab === 'results' ? 'active' : ''}`} 
                 href="#" onClick={() => setActiveTab('results')}>Results</a>
            </li>
            <li className="nav-item">
              <a className={`nav-link ${activeTab === 'settings' ? 'active' : ''}`} 
                 href="#" onClick={() => setActiveTab('settings')}>Settings</a>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
  
  // Simple Dashboard component
  const SimpleDashboard = () => (
    <div>
      <h1 className="mb-4">Dashboard</h1>
      <div className="row">
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">Recent Backtests</div>
            <div className="card-body">
              {backtests.length > 0 ? (
                <ul className="list-group">
                  {backtests.map(backtest => (
                    <li key={backtest.id} className="list-group-item d-flex justify-content-between align-items-center">
                      {backtest.strategy_type} - {backtest.symbols.join(', ')}
                      <span className={`badge ${backtest.status === 'completed' ? 'bg-success' : 'bg-warning'}`}>
                        {backtest.status}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No recent backtests</p>
              )}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">API Status</div>
            <div className="card-body">
              <div className="d-flex align-items-center mb-2">
                <span className="status-indicator status-connected"></span>
                <span>Angel One API: Connected</span>
              </div>
              <div className="d-flex align-items-center mb-2">
                <span className="status-indicator status-connected"></span>
                <span>Client ID: AAAM356344</span>
              </div>
              <div className="d-flex align-items-center">
                <span className="status-indicator status-connected"></span>
                <span>Data Feed: Active</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
  
  // Render active component based on tab
  const renderActiveComponent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <SimpleDashboard />;
      case 'strategies':
        return <div><h1>Strategies</h1><p>Strategy configuration will be displayed here.</p></div>;
      case 'backtest':
        return <div><h1>Backtest</h1><p>Backtest configuration will be displayed here.</p></div>;
      case 'livesimulation':
        return <div><h1>Live Simulation</h1><p>Live simulation configuration will be displayed here.</p></div>;
      case 'results':
        return <div><h1>Results</h1><p>Backtest and simulation results will be displayed here.</p></div>;
      case 'settings':
        return <div><h1>Settings</h1><p>System settings will be displayed here.</p></div>;
      default:
        return <SimpleDashboard />;
    }
  };
  
  return (
    <div className="container-fluid p-0">
      <SimpleNavbar />
      
      <div className="container mt-4">
        {loading ? (
          <div className="d-flex justify-content-center">
            <div className="spinner-border text-primary" role="status">
              <span className="visually-hidden">Loading...</span>
            </div>
          </div>
        ) : error ? (
          <div className="alert alert-danger" role="alert">
            {error}
          </div>
        ) : (
          renderActiveComponent()
        )}
      </div>
    </div>
  );
};

// Render the App
ReactDOM.createRoot(document.getElementById('app')).render(<App />);
