// Navbar Component
const Navbar = () => {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container-fluid">
        <a className="navbar-brand" href="#">
          <i className="bi bi-graph-up-arrow me-2"></i>
          KITE Trading System
        </a>
        <button 
          className="navbar-toggler" 
          type="button" 
          data-bs-toggle="collapse" 
          data-bs-target="#navbarNav" 
          aria-controls="navbarNav" 
          aria-expanded="false" 
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav ms-auto">
            <li className="nav-item">
              <a className="nav-link" href="#" data-bs-toggle="modal" data-bs-target="#apiStatusModal">
                <i className="bi bi-wifi me-1"></i>
                API Status
              </a>
            </li>
            <li className="nav-item">
              <a className="nav-link" href="#">
                <i className="bi bi-question-circle me-1"></i>
                Help
              </a>
            </li>
          </ul>
        </div>
      </div>
      
      {/* API Status Modal */}
      <div className="modal fade" id="apiStatusModal" tabIndex="-1" aria-labelledby="apiStatusModalLabel" aria-hidden="true">
        <div className="modal-dialog">
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title" id="apiStatusModalLabel">API Connection Status</h5>
              <button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <div className="modal-body">
              <div className="card mb-3">
                <div className="card-body">
                  <h6 className="card-title">Angel One Trading API</h6>
                  <div className="d-flex justify-content-between align-items-center">
                    <span>Status:</span>
                    <span className="badge bg-success">Connected</span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center mt-2">
                    <span>Client ID:</span>
                    <span>AAAM356344</span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center mt-2">
                    <span>API Key:</span>
                    <span>ZNQY5zne</span>
                  </div>
                </div>
              </div>
              
              <div className="card mb-3">
                <div className="card-body">
                  <h6 className="card-title">Historical Data API</h6>
                  <div className="d-flex justify-content-between align-items-center">
                    <span>Status:</span>
                    <span className="badge bg-success">Connected</span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center mt-2">
                    <span>API Key:</span>
                    <span>10XN79Ba</span>
                  </div>
                </div>
              </div>
              
              <div className="card">
                <div className="card-body">
                  <h6 className="card-title">Market Feeds API</h6>
                  <div className="d-flex justify-content-between align-items-center">
                    <span>Status:</span>
                    <span className="badge bg-success">Connected</span>
                  </div>
                  <div className="d-flex justify-content-between align-items-center mt-2">
                    <span>API Key:</span>
                    <span>nf3HXMX1</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" data-bs-dismiss="modal">Close</button>
              <button type="button" className="btn btn-primary">Refresh Status</button>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};
