import React from 'react';
import { Form, Button, Card, Row, Col } from 'react-bootstrap';

const PaperSearch = ({ searchQuery, setSearchQuery, onSubmit, loading }) => {
  return (
    <Card className="search-card shadow-sm">
      <Card.Body>
        <div className="text-center mb-4">
          <h3 className="search-title">
            <i className="bi bi-search me-2"></i>
            Search arXiv Research Papers
          </h3>
          <p className="text-muted">
            Enter keywords to find relevant research papers from arXiv
          </p>
        </div>

        <Form onSubmit={onSubmit}>
          <Row>
            <Col xs={12} md={8} className="mx-auto">
              <Form.Group className="mb-3">
                <Form.Label className="fw-semibold">
                  Research Topic or Keywords
                </Form.Label>
                <Form.Control
                  type="text"
                  placeholder="e.g., machine learning, natural language processing, computer vision..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  disabled={loading}
                  className="form-control-lg"
                />
                <Form.Text className="text-muted">
                  Try using specific terms like "transformer neural networks" or "deep learning optimization"
                </Form.Text>
              </Form.Group>

              <div className="d-grid">
                <Button 
                  variant="primary" 
                  size="lg" 
                  type="submit" 
                  disabled={loading || !searchQuery.trim()}
                  className="search-btn"
                >
                  {loading ? (
                    <>
                      <span 
                        className="spinner-border spinner-border-sm me-2" 
                        role="status" 
                        aria-hidden="true"
                      ></span>
                      Searching arXiv...
                    </>
                  ) : (
                    <>
                      <i className="bi bi-search me-2"></i>
                      Search Papers
                    </>
                  )}
                </Button>
              </div>
            </Col>
          </Row>
        </Form>

        {/* Search tips */}
        <div className="search-tips mt-4">
          <h6 className="tips-title">
            <i className="bi bi-lightbulb me-2"></i>
            Search Tips:
          </h6>
          <Row>
            <Col md={6}>
              <ul className="tips-list">
                <li>Use specific technical terms</li>
                <li>Try multiple keywords together</li>
                <li>Include field names (e.g., "NLP", "CV")</li>
              </ul>
            </Col>
            <Col md={6}>
              <ul className="tips-list">
                <li>Avoid very broad terms</li>
                <li>Use quotes for exact phrases</li>
                <li>Try different keyword combinations</li>
              </ul>
            </Col>
          </Row>
        </div>

        {/* Popular search examples */}
        <div className="popular-searches mt-3">
          <h6 className="examples-title">
            <i className="bi bi-fire me-2"></i>
            Popular Searches:
          </h6>
          <div className="example-buttons">
            {[
              'transformer neural networks',
              'computer vision deep learning',
              'natural language processing',
              'reinforcement learning algorithms',
              'graph neural networks',
              'federated learning privacy'
            ].map((example, index) => (
              <Button
                key={index}
                variant="outline-secondary"
                size="sm"
                className="example-btn me-2 mb-2"
                onClick={() => setSearchQuery(example)}
                disabled={loading}
              >
                {example}
              </Button>
            ))}
          </div>
        </div>
      </Card.Body>
    </Card>
  );
};

export default PaperSearch;