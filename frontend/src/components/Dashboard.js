// Dashboard Component
const Dashboard = ({ backtests, simulations }) => {
  // Calculate summary metrics
  const calculateSummaryMetrics = () => {
    const metrics = {
      totalBacktests: backtests.length,
      activeSimulations: simulations.filter(s => s.status === 'running').length,
      completedBacktests: backtests.filter(b => b.status === 'completed').length,
      bestReturn: 0,
      avgWinRate: 0
    };
    
    // Calculate best return and average win rate from completed backtests
    const completedBacktestsWithMetrics = backtests.filter(b => b.status === 'completed' && b.metrics);
    
    if (completedBacktestsWithMetrics.length > 0) {
      metrics.bestReturn = Math.max(...completedBacktestsWithMetrics.map(b => b.metrics.total_return || 0));
      
      const totalWinRate = completedBacktestsWithMetrics.reduce((sum, b) => sum + (b.metrics.win_rate || 0), 0);
      metrics.avgWinRate = totalWinRate / completedBacktestsWithMetrics.length;
    }
    
    return metrics;
  };
  
  const summaryMetrics = calculateSummaryMetrics();
  
  // Get recent backtests (last 5)
  const recentBacktests = [...backtests]
    .sort((a, b) => new Date(b.end_date) - new Date(a.end_date))
    .slice(0, 5);
  
  // Get active simulations
  const activeSimulations = simulations.filter(s => s.status === 'running');
  
  return (
    <div>
      <h1 className="mb-4">Dashboard</h1>
      
      {/* Summary Section */}
      <div className="dashboard-summary mb-4">
        <h2>Trading Summary</h2>
        <div className="metrics-row">
          <div className="metric-item">
            <div className="metric-value">{summaryMetrics.totalBacktests}</div>
            <div className="metric-label">Total Backtests</div>
          </div>
          <div className="metric-item">
            <div className="metric-value">{summaryMetrics.activeSimulations}</div>
            <div className="metric-label">Active Simulations</div>
          </div>
          <div className="metric-item">
            <div className="metric-value">{summaryMetrics.bestReturn.toFixed(2)}%</div>
            <div className="metric-label">Best Return</div>
          </div>
          <div className="metric-item">
            <div className="metric-value">{summaryMetrics.avgWinRate.toFixed(2)}%</div>
            <div className="metric-label">Avg Win Rate</div>
          </div>
        </div>
      </div>
      
      <div className="row">
        {/* Recent Backtests */}
        <div className="col-md-6">
          <div className="card">
            <div className="card-header d-flex justify-content-between align-items-center">
              <h5 className="mb-0">Recent Backtests</h5>
              <button className="btn btn-sm btn-outline-primary">View All</button>
            </div>
            <div className="card-body">
              {recentBacktests.length > 0 ? (
                <div className="table-responsive">
                  <table className="table table-hover">
                    <thead>
                      <tr>
                        <th>Strategy</th>
                        <th>Symbols</th>
                        <th>Period</th>
                        <th>Status</th>
                        <th>Return</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentBacktests.map(backtest => (
                        <tr key={backtest.id}>
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
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-4">
                  <p className="text-muted">No recent backtests found</p>
                  <button className="btn btn-primary">Run a Backtest</button>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Active Simulations */}
        <div className="col-md-6">
          <div className="card">
            <div className="card-header d-flex justify-content-between align-items-center">
              <h5 className="mb-0">Active Simulations</h5>
              <button className="btn btn-sm btn-outline-primary">Start New</button>
            </div>
            <div className="card-body">
              {activeSimulations.length > 0 ? (
                <div className="table-responsive">
                  <table className="table table-hover">
                    <thead>
                      <tr>
                        <th>Strategy</th>
                        <th>Symbols</th>
                        <th>Started</th>
                        <th>Current Return</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeSimulations.map(simulation => (
                        <tr key={simulation.id}>
                          <td>{simulation.strategy_type}</td>
                          <td>{simulation.symbols.join(', ')}</td>
                          <td>{new Date(simulation.start_time).toLocaleString()}</td>
                          <td>
                            {simulation.metrics && simulation.metrics.total_return 
                              ? `${simulation.metrics.total_return.toFixed(2)}%` 
                              : '-'}
                          </td>
                          <td>
                            <button className="btn btn-sm btn-danger">
                              <i className="bi bi-stop-fill"></i> Stop
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-4">
                  <p className="text-muted">No active simulations</p>
                  <button className="btn btn-primary">Start Simulation</button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Quick Actions */}
      <div className="row mt-4">
        <div className="col-12">
          <div className="card">
            <div className="card-header">
              <h5 className="mb-0">Quick Actions</h5>
            </div>
            <div className="card-body">
              <div className="d-flex flex-wrap gap-2">
                <button className="btn btn-primary">
                  <i className="bi bi-clock-history me-2"></i>
                  Run Backtest
                </button>
                <button className="btn btn-success">
                  <i className="bi bi-play-fill me-2"></i>
                  Start Simulation
                </button>
                <button className="btn btn-info">
                  <i className="bi bi-gear me-2"></i>
                  Configure Strategy
                </button>
                <button className="btn btn-warning">
                  <i className="bi bi-bar-chart me-2"></i>
                  View Results
                </button>
                <button className="btn btn-secondary">
                  <i className="bi bi-download me-2"></i>
                  Export Data
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
