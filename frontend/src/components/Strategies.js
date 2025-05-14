// Strategies Component
const Strategies = ({ strategies }) => {
  return (
    <div>
      <h1 className="mb-4">Trading Strategies</h1>
      
      <div className="row">
        {strategies.map(strategy => (
          <div className="col-md-6 mb-4" key={strategy.id}>
            <div className="card h-100">
              <div className="card-header d-flex justify-content-between align-items-center">
                <h5 className="mb-0">{strategy.name}</h5>
                <span className="badge bg-success">Available</span>
              </div>
              <div className="card-body">
                <p className="card-text">{strategy.description}</p>
                
                <h6 className="mt-4 mb-3">Parameters</h6>
                <div className="table-responsive">
                  <table className="table table-sm">
                    <thead>
                      <tr>
                        <th>Parameter</th>
                        <th>Default</th>
                        <th>Range</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(strategy.parameters).map(([key, param]) => (
                        <tr key={key}>
                          <td>{key}</td>
                          <td>{param.default}</td>
                          <td>{param.min} - {param.max}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="card-footer">
                <div className="d-grid gap-2 d-md-flex justify-content-md-end">
                  <button className="btn btn-primary">
                    <i className="bi bi-clock-history me-2"></i>
                    Run Backtest
                  </button>
                  <button className="btn btn-success">
                    <i className="bi bi-play-fill me-2"></i>
                    Start Simulation
                  </button>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      <div className="card mt-4">
        <div className="card-header">
          <h5 className="mb-0">Create Custom Strategy</h5>
        </div>
        <div className="card-body">
          <p>You can create your own custom trading strategy by implementing the Strategy interface.</p>
          <p>Custom strategies should be placed in the <code>kite/strategies/custom</code> directory.</p>
          <button className="btn btn-outline-primary">
            <i className="bi bi-file-earmark-code me-2"></i>
            View Documentation
          </button>
        </div>
      </div>
    </div>
  );
};
