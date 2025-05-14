// Results Component
const Results = ({ backtests, simulations }) => {
  // State for selected result
  const [selectedResult, setSelectedResult] = React.useState(null);
  const [resultType, setResultType] = React.useState(null);
  const [resultData, setResultData] = React.useState(null);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);
  
  // Chart references
  const equityChartRef = React.useRef(null);
  const equityChart = React.useRef(null);
  const drawdownChartRef = React.useRef(null);
  const drawdownChart = React.useRef(null);
  
  // Load result data
  const loadResultData = async (id, type) => {
    try {
      setLoading(true);
      setError(null);
      
      console.log(`Loading result data for ${type} with ID: ${id}`);
      
      // Fetch equity curve data
      console.log(`Fetching equity data from: /api/results/${id}/equity.csv`);
      const equityResponse = await fetch(`/api/results/${id}/equity.csv`);
      console.log('Equity response status:', equityResponse.status, equityResponse.statusText);
      
      if (!equityResponse.ok) {
        throw new Error(`Failed to load equity data: ${equityResponse.status} ${equityResponse.statusText}`);
      }
      
      // Parse CSV data
      const equityText = await equityResponse.text();
      console.log('Received equity data:', equityText.substring(0, 200) + '...');
      
      // Parse CSV
      const rows = equityText.trim().split('\n');
      console.log(`CSV has ${rows.length} rows`);
      
      if (rows.length === 0) {
        throw new Error('CSV data is empty');
      }
      
      const headers = rows[0].split(',');
      console.log('CSV headers:', headers);
      
      const equityData = {
        timestamps: [],
        equity: [],
        drawdown: []
      };
      
      for (let i = 1; i < rows.length; i++) {
        if (rows[i].trim() === '') continue;
        
        const values = rows[i].split(',');
        const rowData = {};
        
        headers.forEach((header, index) => {
          rowData[header.trim()] = values[index];
        });
        
        equityData.timestamps.push(new Date(rowData.timestamp || rowData.date || rowData.index));
        equityData.equity.push(parseFloat(rowData.equity || rowData.portfolio_value));
        
        if (rowData.drawdown) {
          equityData.drawdown.push(parseFloat(rowData.drawdown) * 100);
        }
      }
      
      // Fetch metrics
      const metricsResponse = await fetch(`/api/results/${id}/metrics.json`);
      if (!metricsResponse.ok) {
        throw new Error(`Failed to load metrics: ${metricsResponse.statusText}`);
      }
      const metrics = await metricsResponse.json();
      
      // Try to fetch trades
      let trades = [];
      try {
        console.log(`Fetching trades data from: /api/results/${id}/trades.json`);
        const tradesResponse = await fetch(`/api/results/${id}/trades.json`);
        console.log('Trades response status:', tradesResponse.status, tradesResponse.statusText);
        
        if (tradesResponse.ok) {
          trades = await tradesResponse.json();
          console.log(`Loaded ${trades.length} trades`);
        }
      } catch (e) {
        console.warn('No trades data available:', e);
      }
      
      // Set result data
      setResultData({
        id,
        type,
        equityData,
        metrics,
        trades
      });
      
      // Create charts
      setTimeout(() => {
        createCharts(equityData);
      }, 100);
      
    } catch (err) {
      console.error('Error loading result data:', err);
      setError(`Failed to load result data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Create charts
  const createCharts = (equityData) => {
    // Destroy existing charts
    if (equityChart.current) {
      equityChart.current.destroy();
    }
    
    if (drawdownChartRef.current && drawdownChartRef.current.chart) {
      drawdownChartRef.current.chart.destroy();
    }
    
    // Create equity chart
    if (equityChartRef.current) {
      const ctx = equityChartRef.current.getContext('2d');
      
      equityChart.current = new Chart(ctx, {
        type: 'line',
        data: {
          labels: equityData.timestamps,
          datasets: [{
            label: 'Equity',
            data: equityData.equity,
            borderColor: '#2c3e50',
            backgroundColor: 'rgba(44, 62, 80, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              type: 'time',
              time: {
                unit: 'day'
              },
              title: {
                display: true,
                text: 'Date'
              }
            },
            y: {
              title: {
                display: true,
                text: 'Equity ($)'
              }
            }
          },
          plugins: {
            title: {
              display: true,
              text: 'Equity Curve'
            },
            tooltip: {
              mode: 'index',
              intersect: false
            }
          }
        }
      });
    }
    
    // Create drawdown chart if data exists
    if (drawdownChartRef.current && equityData.drawdown && equityData.drawdown.length > 0) {
      const ctx = drawdownChartRef.current.getContext('2d');
      
      drawdownChartRef.current.chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: equityData.timestamps,
          datasets: [{
            label: 'Drawdown',
            data: equityData.drawdown,
            borderColor: '#e74c3c',
            backgroundColor: 'rgba(231, 76, 60, 0.1)',
            borderWidth: 2,
            fill: true
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              type: 'time',
              time: {
                unit: 'day'
              },
              title: {
                display: true,
                text: 'Date'
              }
            },
            y: {
              title: {
                display: true,
                text: 'Drawdown (%)'
              },
              reverse: true
            }
          },
          plugins: {
            title: {
              display: true,
              text: 'Drawdown'
            },
            tooltip: {
              mode: 'index',
              intersect: false
            }
          }
        }
      });
    }
  };
  
  // Handle result selection
  const handleResultSelect = (result, type) => {
    setSelectedResult(result);
    setResultType(type);
    loadResultData(result.id, type);
  };
  
  // Format metrics for display
  const formatMetric = (value) => {
    if (typeof value === 'number') {
      // Check if it's a percentage
      if (value > -1 && value < 1) {
        return `${(value * 100).toFixed(2)}%`;
      }
      // Check if it's a large number
      if (value > 1000) {
        return value.toLocaleString('en-US', { maximumFractionDigits: 2 });
      }
      return value.toFixed(4);
    }
    return value;
  };
  
  // Get all results
  const allResults = [
    ...backtests.map(b => ({ ...b, resultType: 'backtest' })),
    ...simulations.map(s => ({ ...s, resultType: 'simulation' }))
  ].sort((a, b) => {
    // Sort by date (newest first)
    const dateA = a.resultType === 'backtest' ? new Date(a.end_date) : new Date(a.start_time);
    const dateB = b.resultType === 'backtest' ? new Date(b.end_date) : new Date(b.start_time);
    return dateB - dateA;
  });
  
  return (
    <div>
      <h1 className="mb-4">Results</h1>
      
      {/* Alert for errors */}
      {error && (
        <div className="alert alert-danger alert-dismissible fade show" role="alert">
          {error}
          <button type="button" className="btn-close" onClick={() => setError(null)}></button>
        </div>
      )}
      
      <div className="row">
        {/* Results List */}
        <div className="col-md-4">
          <div className="card">
            <div className="card-header d-flex justify-content-between align-items-center">
              <h5 className="mb-0">Available Results</h5>
              <div className="btn-group btn-group-sm" role="group">
                <button type="button" className="btn btn-outline-primary">All</button>
                <button type="button" className="btn btn-outline-primary">Backtests</button>
                <button type="button" className="btn btn-outline-primary">Simulations</button>
              </div>
            </div>
            <div className="card-body p-0">
              {allResults.length > 0 ? (
                <div className="list-group list-group-flush">
                  {allResults.map(result => (
                    <a 
                      href="#" 
                      className={`list-group-item list-group-item-action ${selectedResult && selectedResult.id === result.id ? 'active' : ''}`}
                      key={result.id}
                      onClick={() => handleResultSelect(result, result.resultType)}
                    >
                      <div className="d-flex w-100 justify-content-between">
                        <h6 className="mb-1">
                          {result.strategy_type.toUpperCase()} - {result.symbols.join(', ')}
                        </h6>
                        <small>
                          <span className={`badge ${
                            result.status === 'completed' ? 'bg-success' : 
                            result.status === 'running' ? 'bg-primary' : 
                            result.status === 'failed' ? 'bg-danger' : 'bg-secondary'
                          }`}>
                            {result.status}
                          </span>
                        </small>
                      </div>
                      <p className="mb-1">
                        {result.resultType === 'backtest' 
                          ? `${result.start_date} to ${result.end_date}` 
                          : `Started: ${new Date(result.start_time).toLocaleDateString()}`
                        }
                      </p>
                      <small>
                        {result.metrics && result.metrics.total_return 
                          ? `Return: ${formatMetric(result.metrics.total_return)}` 
                          : 'No metrics available'}
                      </small>
                    </a>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4">
                  <p className="text-muted">No results available</p>
                  <button className="btn btn-primary">Run a Backtest</button>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Result Details */}
        <div className="col-md-8">
          {selectedResult ? (
            <div>
              {loading ? (
                <div className="d-flex justify-content-center my-5">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                </div>
              ) : resultData ? (
                <div>
                  {/* Result Header */}
                  <div className="card mb-4">
                    <div className="card-header">
                      <h5 className="mb-0">
                        {selectedResult.strategy_type.toUpperCase()} - {selectedResult.symbols.join(', ')}
                      </h5>
                    </div>
                    <div className="card-body">
                      <div className="row">
                        <div className="col-md-6">
                          <p><strong>Type:</strong> {resultType === 'backtest' ? 'Backtest' : 'Live Simulation'}</p>
                          <p>
                            <strong>Period:</strong> {resultType === 'backtest' 
                              ? `${selectedResult.start_date} to ${selectedResult.end_date}` 
                              : `Started: ${new Date(selectedResult.start_time).toLocaleDateString()}`
                            }
                          </p>
                          <p><strong>Status:</strong> {selectedResult.status}</p>
                        </div>
                        <div className="col-md-6">
                          <p><strong>ID:</strong> {selectedResult.id}</p>
                          <p><strong>Symbols:</strong> {selectedResult.symbols.join(', ')}</p>
                          <p>
                            <strong>Strategy:</strong> {selectedResult.strategy_type}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Performance Metrics */}
                  <div className="card mb-4">
                    <div className="card-header">
                      <h5 className="mb-0">Performance Metrics</h5>
                    </div>
                    <div className="card-body">
                      <div className="row">
                        {resultData.metrics && Object.entries(resultData.metrics).map(([key, value]) => (
                          <div className="col-md-3 mb-3" key={key}>
                            <div className="metrics-card">
                              <div className="metrics-value">{formatMetric(value)}</div>
                              <div className="metrics-label">{key.replace(/_/g, ' ').toUpperCase()}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                  
                  {/* Charts */}
                  <div className="row">
                    <div className="col-md-12 mb-4">
                      <div className="card">
                        <div className="card-header">
                          <h5 className="mb-0">Equity Curve</h5>
                        </div>
                        <div className="card-body">
                          <div className="chart-container">
                            <canvas ref={equityChartRef}></canvas>
                          </div>
                        </div>
                      </div>
                    </div>
                    
                    <div className="col-md-12 mb-4">
                      <div className="card">
                        <div className="card-header">
                          <h5 className="mb-0">Drawdown</h5>
                        </div>
                        <div className="card-body">
                          <div className="chart-container">
                            <canvas ref={drawdownChartRef}></canvas>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Trades */}
                  {resultData.trades && resultData.trades.length > 0 && (
                    <div className="card mb-4">
                      <div className="card-header d-flex justify-content-between align-items-center">
                        <h5 className="mb-0">Trades</h5>
                        <button className="btn btn-sm btn-outline-primary">
                          <i className="bi bi-download me-1"></i>
                          Export
                        </button>
                      </div>
                      <div className="card-body">
                        <div className="table-responsive">
                          <table className="table table-sm table-hover trade-table">
                            <thead>
                              <tr>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Quantity</th>
                                <th>Price</th>
                                <th>Timestamp</th>
                                <th>P&L</th>
                              </tr>
                            </thead>
                            <tbody>
                              {resultData.trades.slice(0, 10).map((trade, index) => (
                                <tr key={index}>
                                  <td>{trade.symbol}</td>
                                  <td>
                                    <span className={`badge ${trade.side === 'BUY' ? 'bg-success' : 'bg-danger'}`}>
                                      {trade.side}
                                    </span>
                                  </td>
                                  <td>{trade.quantity}</td>
                                  <td>${parseFloat(trade.price).toFixed(2)}</td>
                                  <td>{new Date(trade.timestamp).toLocaleString()}</td>
                                  <td>
                                    {trade.pnl !== undefined ? (
                                      <span className={trade.pnl >= 0 ? 'text-success' : 'text-danger'}>
                                        ${parseFloat(trade.pnl).toFixed(2)}
                                      </span>
                                    ) : '-'}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        {resultData.trades.length > 10 && (
                          <div className="text-center mt-3">
                            <button className="btn btn-sm btn-outline-secondary">
                              View All {resultData.trades.length} Trades
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {/* Actions */}
                  <div className="d-flex justify-content-end mb-4">
                    <button className="btn btn-outline-secondary me-2">
                      <i className="bi bi-download me-1"></i>
                      Export Results
                    </button>
                    <button className="btn btn-primary">
                      <i className="bi bi-arrow-repeat me-1"></i>
                      Run Again
                    </button>
                  </div>
                </div>
              ) : (
                <div className="card">
                  <div className="card-body text-center py-5">
                    <p className="text-muted">Select a result to view details</p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="card">
              <div className="card-body text-center py-5">
                <h5 className="mb-3">No Result Selected</h5>
                <p className="text-muted">Select a result from the list to view details</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
