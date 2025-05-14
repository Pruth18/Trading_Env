// LiveSimulation Component
const LiveSimulation = ({ strategies, symbols, simulations, setSimulations }) => {
  // State for form
  const [formData, setFormData] = React.useState({
    strategy_type: 'rsi',
    symbols: ['RELIANCE'],
    interval: '1d',
    initial_capital: 100000,
    commission_rate: 0.002,
    slippage_rate: 0.0005,
    history_days: 180,
    update_interval: 60,
    rsi_params: {
      rsi_period: 8,
      oversold_level: 40,
      overbought_level: 60,
      position_size: 0.2
    },
    ma_params: {
      fast_period: 20,
      slow_period: 50,
      signal_period: 9,
      position_size: 0.15
    }
  });
  
  // State for loading
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [success, setSuccess] = React.useState(null);
  
  // Handle form input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Handle strategy parameter changes
  const handleParamChange = (strategyType, paramName, value) => {
    setFormData(prev => ({
      ...prev,
      [`${strategyType}_params`]: {
        ...prev[`${strategyType}_params`],
        [paramName]: typeof value === 'string' ? parseFloat(value) : value
      }
    }));
  };
  
  // Handle symbol selection
  const handleSymbolChange = (e) => {
    const options = e.target.options;
    const selectedSymbols = [];
    for (let i = 0; i < options.length; i++) {
      if (options[i].selected) {
        selectedSymbols.push(options[i].value);
      }
    }
    setFormData(prev => ({
      ...prev,
      symbols: selectedSymbols
    }));
  };
  
  // Handle strategy selection
  const handleStrategySelect = (strategyType) => {
    setFormData(prev => ({
      ...prev,
      strategy_type: strategyType
    }));
  };
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      setLoading(true);
      setError(null);
      setSuccess(null);
      
      // Prepare request payload
      const payload = {
        strategy_type: formData.strategy_type,
        symbols: formData.symbols,
        interval: formData.interval,
        initial_capital: parseFloat(formData.initial_capital),
        commission_rate: parseFloat(formData.commission_rate),
        slippage_rate: parseFloat(formData.slippage_rate),
        history_days: parseInt(formData.history_days),
        update_interval: parseInt(formData.update_interval)
      };
      
      // Add strategy-specific parameters
      if (formData.strategy_type === 'rsi') {
        payload.rsi_params = formData.rsi_params;
      } else if (formData.strategy_type === 'ma_crossover') {
        payload.ma_params = formData.ma_params;
      }
      
      // Send request to API
      const response = await fetch('/api/simulation', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Update simulations state
      if (setSimulations) {
        setSimulations(prev => [data, ...prev]);
      }
      
      setSuccess('Simulation started successfully!');
      
    } catch (err) {
      console.error('Error starting simulation:', err);
      setError(`Failed to start simulation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Function to stop a simulation
  const handleStopSimulation = async (simulationId) => {
    try {
      const response = await fetch(`/api/simulation/${simulationId}/stop`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      // Update simulation status
      if (setSimulations) {
        setSimulations(prev => 
          prev.map(sim => 
            sim.id === simulationId 
              ? { ...sim, status: 'stopped' } 
              : sim
          )
        );
      }
      
      setSuccess(`Simulation ${simulationId} stopped successfully!`);
      
    } catch (err) {
      console.error('Error stopping simulation:', err);
      setError(`Failed to stop simulation: ${err.message}`);
    }
  };
  
  return (
    <div>
      <h1 className="mb-4">Live Simulation</h1>
      
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
        {/* Strategy Selection */}
        <div className="col-md-4">
          <div className="card mb-4">
            <div className="card-header">
              <h5 className="mb-0">Select Strategy</h5>
            </div>
            <div className="card-body">
              <div className="row">
                {strategies.map(strategy => (
                  <div className="col-md-12 mb-3" key={strategy.id}>
                    <div 
                      className={`card strategy-card ${formData.strategy_type === strategy.id ? 'border-primary' : ''}`}
                      onClick={() => handleStrategySelect(strategy.id)}
                    >
                      <div className="card-body">
                        <h5 className="card-title">
                          {strategy.name}
                          {formData.strategy_type === strategy.id && (
                            <i className="bi bi-check-circle-fill text-primary ms-2"></i>
                          )}
                        </h5>
                        <p className="card-text">{strategy.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
        
        {/* Simulation Configuration */}
        <div className="col-md-8">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Configure Simulation</h5>
            </div>
            <div className="card-body">
              <form onSubmit={handleSubmit}>
                {/* Symbols */}
                <div className="mb-3">
                  <label htmlFor="symbols" className="form-label">Symbols</label>
                  <select 
                    className="form-select" 
                    id="symbols" 
                    name="symbols" 
                    multiple 
                    value={formData.symbols}
                    onChange={handleSymbolChange}
                    required
                  >
                    {symbols.map(symbol => (
                      <option key={symbol.symbol} value={symbol.symbol}>
                        {symbol.symbol} - {symbol.name}
                      </option>
                    ))}
                  </select>
                  <div className="form-text">Hold Ctrl/Cmd to select multiple symbols</div>
                </div>
                
                <div className="row">
                  {/* Interval */}
                  <div className="col-md-4">
                    <div className="mb-3">
                      <label htmlFor="interval" className="form-label">Interval</label>
                      <select 
                        className="form-select" 
                        id="interval" 
                        name="interval"
                        value={formData.interval}
                        onChange={handleInputChange}
                      >
                        <option value="1m">1 Minute</option>
                        <option value="5m">5 Minutes</option>
                        <option value="15m">15 Minutes</option>
                        <option value="30m">30 Minutes</option>
                        <option value="1h">1 Hour</option>
                        <option value="1d">1 Day</option>
                        <option value="1w">1 Week</option>
                      </select>
                    </div>
                  </div>
                  
                  {/* Initial Capital */}
                  <div className="col-md-4">
                    <div className="mb-3">
                      <label htmlFor="initial_capital" className="form-label">Initial Capital</label>
                      <input 
                        type="number" 
                        className="form-control" 
                        id="initial_capital" 
                        name="initial_capital"
                        value={formData.initial_capital}
                        onChange={handleInputChange}
                        min="1000"
                        step="1000"
                        required
                      />
                    </div>
                  </div>
                  
                  {/* Commission Rate */}
                  <div className="col-md-4">
                    <div className="mb-3">
                      <label htmlFor="commission_rate" className="form-label">Commission Rate</label>
                      <input 
                        type="number" 
                        className="form-control" 
                        id="commission_rate" 
                        name="commission_rate"
                        value={formData.commission_rate}
                        onChange={handleInputChange}
                        min="0"
                        max="0.1"
                        step="0.001"
                        required
                      />
                      <div className="form-text">e.g., 0.002 = 0.2%</div>
                    </div>
                  </div>
                </div>
                
                <div className="row">
                  {/* History Days */}
                  <div className="col-md-6">
                    <div className="mb-3">
                      <label htmlFor="history_days" className="form-label">History Days</label>
                      <input 
                        type="number" 
                        className="form-control" 
                        id="history_days" 
                        name="history_days"
                        value={formData.history_days}
                        onChange={handleInputChange}
                        min="30"
                        max="365"
                        required
                      />
                      <div className="form-text">Days of historical data to initialize with</div>
                    </div>
                  </div>
                  
                  {/* Update Interval */}
                  <div className="col-md-6">
                    <div className="mb-3">
                      <label htmlFor="update_interval" className="form-label">Update Interval (seconds)</label>
                      <input 
                        type="number" 
                        className="form-control" 
                        id="update_interval" 
                        name="update_interval"
                        value={formData.update_interval}
                        onChange={handleInputChange}
                        min="10"
                        max="3600"
                        required
                      />
                      <div className="form-text">Interval between updates in seconds</div>
                    </div>
                  </div>
                </div>
                
                {/* Strategy-specific parameters */}
                {formData.strategy_type === 'rsi' && (
                  <div className="card mt-3">
                    <div className="card-header">
                      <h6 className="mb-0">RSI Strategy Parameters</h6>
                    </div>
                    <div className="card-body">
                      <div className="row">
                        <div className="col-md-6">
                          <div className="mb-3">
                            <label htmlFor="rsi_period" className="form-label">RSI Period</label>
                            <input 
                              type="number" 
                              className="form-control" 
                              id="rsi_period" 
                              value={formData.rsi_params.rsi_period}
                              onChange={(e) => handleParamChange('rsi', 'rsi_period', parseInt(e.target.value))}
                              min="2"
                              max="50"
                              required
                            />
                          </div>
                        </div>
                        <div className="col-md-6">
                          <div className="mb-3">
                            <label htmlFor="position_size" className="form-label">Position Size</label>
                            <input 
                              type="number" 
                              className="form-control" 
                              id="position_size" 
                              value={formData.rsi_params.position_size}
                              onChange={(e) => handleParamChange('rsi', 'position_size', e.target.value)}
                              min="0"
                              max="1"
                              step="0.05"
                              required
                            />
                            <div className="form-text">Percentage of portfolio (0-1)</div>
                          </div>
                        </div>
                      </div>
                      <div className="row">
                        <div className="col-md-6">
                          <div className="mb-3">
                            <label htmlFor="oversold_level" className="form-label">Oversold Level</label>
                            <input 
                              type="number" 
                              className="form-control" 
                              id="oversold_level" 
                              value={formData.rsi_params.oversold_level}
                              onChange={(e) => handleParamChange('rsi', 'oversold_level', e.target.value)}
                              min="10"
                              max="50"
                              required
                            />
                          </div>
                        </div>
                        <div className="col-md-6">
                          <div className="mb-3">
                            <label htmlFor="overbought_level" className="form-label">Overbought Level</label>
                            <input 
                              type="number" 
                              className="form-control" 
                              id="overbought_level" 
                              value={formData.rsi_params.overbought_level}
                              onChange={(e) => handleParamChange('rsi', 'overbought_level', e.target.value)}
                              min="50"
                              max="90"
                              required
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {formData.strategy_type === 'ma_crossover' && (
                  <div className="card mt-3">
                    <div className="card-header">
                      <h6 className="mb-0">Moving Average Crossover Parameters</h6>
                    </div>
                    <div className="card-body">
                      <div className="row">
                        <div className="col-md-4">
                          <div className="mb-3">
                            <label htmlFor="fast_period" className="form-label">Fast Period</label>
                            <input 
                              type="number" 
                              className="form-control" 
                              id="fast_period" 
                              value={formData.ma_params.fast_period}
                              onChange={(e) => handleParamChange('ma', 'fast_period', parseInt(e.target.value))}
                              min="2"
                              max="50"
                              required
                            />
                          </div>
                        </div>
                        <div className="col-md-4">
                          <div className="mb-3">
                            <label htmlFor="slow_period" className="form-label">Slow Period</label>
                            <input 
                              type="number" 
                              className="form-control" 
                              id="slow_period" 
                              value={formData.ma_params.slow_period}
                              onChange={(e) => handleParamChange('ma', 'slow_period', parseInt(e.target.value))}
                              min="10"
                              max="200"
                              required
                            />
                          </div>
                        </div>
                        <div className="col-md-4">
                          <div className="mb-3">
                            <label htmlFor="signal_period" className="form-label">Signal Period</label>
                            <input 
                              type="number" 
                              className="form-control" 
                              id="signal_period" 
                              value={formData.ma_params.signal_period}
                              onChange={(e) => handleParamChange('ma', 'signal_period', parseInt(e.target.value))}
                              min="2"
                              max="20"
                              required
                            />
                          </div>
                        </div>
                      </div>
                      <div className="row">
                        <div className="col-md-6">
                          <div className="mb-3">
                            <label htmlFor="ma_position_size" className="form-label">Position Size</label>
                            <input 
                              type="number" 
                              className="form-control" 
                              id="ma_position_size" 
                              value={formData.ma_params.position_size}
                              onChange={(e) => handleParamChange('ma', 'position_size', e.target.value)}
                              min="0"
                              max="1"
                              step="0.05"
                              required
                            />
                            <div className="form-text">Percentage of portfolio (0-1)</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Submit Button */}
                <div className="mt-4 d-grid">
                  <button 
                    type="submit" 
                    className="btn btn-success btn-lg"
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                        Starting Simulation...
                      </>
                    ) : (
                      <>
                        <i className="bi bi-play-fill me-2"></i>
                        Start Simulation
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
      
      {/* Active Simulations */}
      <div className="card mt-4">
        <div className="card-header">
          <h5 className="mb-0">Active Simulations</h5>
        </div>
        <div className="card-body">
          {simulations && simulations.length > 0 ? (
            <div className="table-responsive">
              <table className="table table-hover">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Strategy</th>
                    <th>Symbols</th>
                    <th>Started</th>
                    <th>Status</th>
                    <th>Return</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {simulations.map(simulation => (
                    <tr key={simulation.id}>
                      <td>{simulation.id.substring(0, 8)}...</td>
                      <td>{simulation.strategy_type}</td>
                      <td>{simulation.symbols.join(', ')}</td>
                      <td>{new Date(simulation.start_time).toLocaleString()}</td>
                      <td>
                        <span className={`badge ${
                          simulation.status === 'running' ? 'bg-success' : 
                          simulation.status === 'stopped' ? 'bg-warning' : 
                          simulation.status === 'failed' ? 'bg-danger' : 'bg-secondary'
                        }`}>
                          {simulation.status}
                        </span>
                      </td>
                      <td>
                        {simulation.metrics && simulation.metrics.total_return 
                          ? `${simulation.metrics.total_return.toFixed(2)}%` 
                          : '-'}
                      </td>
                      <td>
                        {simulation.status === 'running' ? (
                          <button 
                            className="btn btn-sm btn-danger"
                            onClick={() => handleStopSimulation(simulation.id)}
                          >
                            <i className="bi bi-stop-fill"></i> Stop
                          </button>
                        ) : (
                          <button className="btn btn-sm btn-info">
                            <i className="bi bi-eye"></i> View
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-muted">No active simulations</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
