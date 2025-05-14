// Backtest Component
const Backtest = ({ strategies, symbols, backtests, setBacktests }) => {
  // State for form
  const [formData, setFormData] = React.useState({
    strategy_type: 'rsi',
    symbols: ['RELIANCE'],
    start_date: new Date(Date.now() - 180 * 24 * 60 * 60 * 1000).toISOString().split('T')[0], // 180 days ago
    end_date: new Date().toISOString().split('T')[0], // today
    interval: '1d',
    initial_capital: 100000,
    commission_rate: 0.002,
    slippage_rate: 0.0005,
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
  
  // State for selected strategy
  const [selectedStrategy, setSelectedStrategy] = React.useState(null);
  
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
  const handleStrategySelect = (strategy) => {
    setSelectedStrategy(strategy);
    setFormData(prev => ({
      ...prev,
      strategy_type: strategy.id
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
        start_date: formData.start_date,
        end_date: formData.end_date,
        interval: formData.interval,
        initial_capital: parseFloat(formData.initial_capital),
        commission_rate: parseFloat(formData.commission_rate),
        slippage_rate: parseFloat(formData.slippage_rate)
      };
      
      // Add strategy-specific parameters
      if (formData.strategy_type === 'rsi') {
        payload.rsi_params = formData.rsi_params;
      } else if (formData.strategy_type === 'ma_crossover') {
        payload.ma_params = formData.ma_params;
      }
      
      // Send request to API
      const response = await fetch('/api/backtest', {
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
      
      // Update backtests state
      setBacktests(prev => [data, ...prev]);
      
      setSuccess('Backtest started successfully!');
      
    } catch (err) {
      console.error('Error starting backtest:', err);
      setError(`Failed to start backtest: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div>
      <h1 className="mb-4">Backtest</h1>
      
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
                      onClick={() => handleStrategySelect(strategy)}
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
        
        {/* Backtest Configuration */}
        <div className="col-md-8">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Configure Backtest</h5>
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
                  {/* Date Range */}
                  <div className="col-md-6">
                    <div className="mb-3">
                      <label htmlFor="start_date" className="form-label">Start Date</label>
                      <input 
                        type="date" 
                        className="form-control" 
                        id="start_date" 
                        name="start_date"
                        value={formData.start_date}
                        onChange={handleInputChange}
                        required
                      />
                    </div>
                  </div>
                  <div className="col-md-6">
                    <div className="mb-3">
                      <label htmlFor="end_date" className="form-label">End Date</label>
                      <input 
                        type="date" 
                        className="form-control" 
                        id="end_date" 
                        name="end_date"
                        value={formData.end_date}
                        onChange={handleInputChange}
                        required
                      />
                    </div>
                  </div>
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
                    className="btn btn-primary btn-lg"
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                        Starting Backtest...
                      </>
                    ) : (
                      <>
                        <i className="bi bi-play-fill me-2"></i>
                        Run Backtest
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
      
      {/* Recent Backtests */}
      <div className="card mt-4">
        <div className="card-header">
          <h5 className="mb-0">Recent Backtests</h5>
        </div>
        <div className="card-body">
          {backtests.length > 0 ? (
            <div className="table-responsive">
              <table className="table table-hover">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Strategy</th>
                    <th>Symbols</th>
                    <th>Period</th>
                    <th>Status</th>
                    <th>Return</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {backtests.map(backtest => (
                    <tr key={backtest.id}>
                      <td>{backtest.id.substring(0, 8)}...</td>
                      <td>{backtest.strategy_type}</td>
                      <td>{backtest.symbols.join(', ')}</td>
                      <td>{backtest.start_date} to {backtest.end_date}</td>
                      <td>
                        <span className={`badge ${
                          backtest.status === 'completed' ? 'bg-success' : 
                          backtest.status === 'running' ? 'bg-primary' : 
                          backtest.status === 'failed' ? 'bg-danger' : 'bg-secondary'
                        }`}>
                          {backtest.status}
                        </span>
                      </td>
                      <td>
                        {backtest.metrics && backtest.metrics.total_return 
                          ? `${backtest.metrics.total_return.toFixed(2)}%` 
                          : '-'}
                      </td>
                      <td>
                        <button className="btn btn-sm btn-info me-1">
                          <i className="bi bi-eye"></i>
                        </button>
                        <button className="btn btn-sm btn-secondary">
                          <i className="bi bi-download"></i>
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-4">
              <p className="text-muted">No backtests found</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
