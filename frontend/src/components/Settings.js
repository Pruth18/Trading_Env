// Settings Component
const Settings = () => {
  // State for API settings
  const [apiSettings, setApiSettings] = React.useState({
    client_id: "AAAM356344",
    trading_api_key: "ZNQY5zne",
    historical_data_api_key: "10XN79Ba",
    market_feeds_api_key: "nf3HXMX1",
    use_real_api: false
  });
  
  // State for general settings
  const [generalSettings, setGeneralSettings] = React.useState({
    default_initial_capital: 100000,
    default_commission_rate: 0.002,
    default_slippage_rate: 0.0005,
    auto_save_results: true,
    dark_mode: false
  });
  
  // State for loading and alerts
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [success, setSuccess] = React.useState(null);
  
  // Handle API settings change
  const handleApiSettingsChange = (e) => {
    const { name, value, type, checked } = e.target;
    setApiSettings(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };
  
  // Handle general settings change
  const handleGeneralSettingsChange = (e) => {
    const { name, value, type, checked } = e.target;
    setGeneralSettings(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : type === 'number' ? parseFloat(value) : value
    }));
  };
  
  // Handle API settings form submission
  const handleApiSettingsSubmit = (e) => {
    e.preventDefault();
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      setSuccess('API settings saved successfully!');
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
    }, 1000);
  };
  
  // Handle general settings form submission
  const handleGeneralSettingsSubmit = (e) => {
    e.preventDefault();
    
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      setSuccess('General settings saved successfully!');
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
    }, 1000);
  };
  
  // Test API connection
  const testApiConnection = () => {
    setLoading(true);
    setError(null);
    setSuccess(null);
    
    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      setSuccess('API connection successful!');
      
      // Clear success message after 3 seconds
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
    }, 1500);
  };
  
  return (
    <div>
      <h1 className="mb-4">Settings</h1>
      
      {/* Alert messages */}
      {error && (
        <div className="alert alert-danger alert-dismissible fade show" role="alert">
          {error}
          <button type="button" className="btn-close" onClick={() => setError(null)}></button>
        </div>
      )}
      
      {success && (
        <div className="alert alert-success alert-dismissible fade show" role="alert">
          {success}
          <button type="button" className="btn-close" onClick={() => setSuccess(null)}></button>
        </div>
      )}
      
      <div className="row">
        {/* API Settings */}
        <div className="col-md-6">
          <div className="card mb-4">
            <div className="card-header">
              <h5 className="mb-0">API Settings</h5>
            </div>
            <div className="card-body">
              <form onSubmit={handleApiSettingsSubmit}>
                <div className="mb-3">
                  <label htmlFor="client_id" className="form-label">Angel One Client ID</label>
                  <input 
                    type="text" 
                    className="form-control" 
                    id="client_id" 
                    name="client_id"
                    value={apiSettings.client_id}
                    onChange={handleApiSettingsChange}
                    required
                  />
                </div>
                
                <div className="mb-3">
                  <label htmlFor="trading_api_key" className="form-label">Trading API Key</label>
                  <input 
                    type="text" 
                    className="form-control" 
                    id="trading_api_key" 
                    name="trading_api_key"
                    value={apiSettings.trading_api_key}
                    onChange={handleApiSettingsChange}
                    required
                  />
                </div>
                
                <div className="mb-3">
                  <label htmlFor="historical_data_api_key" className="form-label">Historical Data API Key</label>
                  <input 
                    type="text" 
                    className="form-control" 
                    id="historical_data_api_key" 
                    name="historical_data_api_key"
                    value={apiSettings.historical_data_api_key}
                    onChange={handleApiSettingsChange}
                    required
                  />
                </div>
                
                <div className="mb-3">
                  <label htmlFor="market_feeds_api_key" className="form-label">Market Feeds API Key</label>
                  <input 
                    type="text" 
                    className="form-control" 
                    id="market_feeds_api_key" 
                    name="market_feeds_api_key"
                    value={apiSettings.market_feeds_api_key}
                    onChange={handleApiSettingsChange}
                    required
                  />
                </div>
                
                <div className="form-check form-switch mb-3">
                  <input 
                    className="form-check-input" 
                    type="checkbox" 
                    id="use_real_api" 
                    name="use_real_api"
                    checked={apiSettings.use_real_api}
                    onChange={handleApiSettingsChange}
                  />
                  <label className="form-check-label" htmlFor="use_real_api">
                    Use Real API (not simulation)
                  </label>
                </div>
                
                <div className="d-flex">
                  <button 
                    type="submit" 
                    className="btn btn-primary me-2"
                    disabled={loading}
                  >
                    {loading ? 'Saving...' : 'Save Settings'}
                  </button>
                  <button 
                    type="button" 
                    className="btn btn-outline-secondary"
                    onClick={testApiConnection}
                    disabled={loading}
                  >
                    {loading ? 'Testing...' : 'Test Connection'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
        
        {/* General Settings */}
        <div className="col-md-6">
          <div className="card mb-4">
            <div className="card-header">
              <h5 className="mb-0">General Settings</h5>
            </div>
            <div className="card-body">
              <form onSubmit={handleGeneralSettingsSubmit}>
                <div className="mb-3">
                  <label htmlFor="default_initial_capital" className="form-label">Default Initial Capital</label>
                  <input 
                    type="number" 
                    className="form-control" 
                    id="default_initial_capital" 
                    name="default_initial_capital"
                    value={generalSettings.default_initial_capital}
                    onChange={handleGeneralSettingsChange}
                    min="1000"
                    step="1000"
                    required
                  />
                </div>
                
                <div className="mb-3">
                  <label htmlFor="default_commission_rate" className="form-label">Default Commission Rate</label>
                  <input 
                    type="number" 
                    className="form-control" 
                    id="default_commission_rate" 
                    name="default_commission_rate"
                    value={generalSettings.default_commission_rate}
                    onChange={handleGeneralSettingsChange}
                    min="0"
                    max="0.1"
                    step="0.001"
                    required
                  />
                  <div className="form-text">e.g., 0.002 = 0.2%</div>
                </div>
                
                <div className="mb-3">
                  <label htmlFor="default_slippage_rate" className="form-label">Default Slippage Rate</label>
                  <input 
                    type="number" 
                    className="form-control" 
                    id="default_slippage_rate" 
                    name="default_slippage_rate"
                    value={generalSettings.default_slippage_rate}
                    onChange={handleGeneralSettingsChange}
                    min="0"
                    max="0.1"
                    step="0.0001"
                    required
                  />
                  <div className="form-text">e.g., 0.0005 = 0.05%</div>
                </div>
                
                <div className="form-check form-switch mb-3">
                  <input 
                    className="form-check-input" 
                    type="checkbox" 
                    id="auto_save_results" 
                    name="auto_save_results"
                    checked={generalSettings.auto_save_results}
                    onChange={handleGeneralSettingsChange}
                  />
                  <label className="form-check-label" htmlFor="auto_save_results">
                    Auto-save Results
                  </label>
                </div>
                
                <div className="form-check form-switch mb-3">
                  <input 
                    className="form-check-input" 
                    type="checkbox" 
                    id="dark_mode" 
                    name="dark_mode"
                    checked={generalSettings.dark_mode}
                    onChange={handleGeneralSettingsChange}
                  />
                  <label className="form-check-label" htmlFor="dark_mode">
                    Dark Mode
                  </label>
                </div>
                
                <button 
                  type="submit" 
                  className="btn btn-primary"
                  disabled={loading}
                >
                  {loading ? 'Saving...' : 'Save Settings'}
                </button>
              </form>
            </div>
          </div>
          
          {/* System Information */}
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">System Information</h5>
            </div>
            <div className="card-body">
              <div className="d-flex justify-content-between mb-2">
                <span>Version:</span>
                <span>1.0.0</span>
              </div>
              <div className="d-flex justify-content-between mb-2">
                <span>Last Updated:</span>
                <span>May 12, 2025</span>
              </div>
              <div className="d-flex justify-content-between mb-2">
                <span>Data Directory:</span>
                <span>C:\Users\Pruthvi\Desktop\KITE\data</span>
              </div>
              <div className="d-flex justify-content-between mb-2">
                <span>Results Directory:</span>
                <span>C:\Users\Pruthvi\Desktop\KITE\results</span>
              </div>
              <div className="d-flex justify-content-between">
                <span>Log Directory:</span>
                <span>C:\Users\Pruthvi\Desktop\KITE\logs</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
